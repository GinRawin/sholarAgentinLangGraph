from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from json import JSONDecodeError
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pypdf import PdfReader

from scholar_agent.utils.state import Paper, SummaryResult


SCHEMA_VERSION = 2


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="sholar_",
        extra="ignore",
    )

    db_path: Path = Field(default=Path("./data/scholar_agent.sqlite3"))
    pdf_root: Path = Field(default=Path("./papers"))
    llm_provider: str = Field(default="placeholder")
    llm_model: str = Field(default="")
    api_key: str = Field(default="")
    base_url: str = Field(default="")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class LLMClient(Protocol):
    def summarize_paper(
        self,
        *,
        title: str,
        prompt: str,
        known_keywords: list[str],
        known_categories: list[str],
        pdf_attachment: PdfAttachment,
    ) -> SummaryResult:
        ...

    def draft_deep_note(
        self,
        *,
        title: str,
        prompt: str,
        related_note_paths: list[str],
        pdf_attachment: PdfAttachment,
    ) -> str:
        ...

    def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
        ...

    def answer_note_question(
        self,
        *,
        title: str,
        current_note: str,
        user_question: str,
    ) -> str:
        ...


@dataclass(slots=True)
class PdfAttachment:
    filename: str
    base64_data: str


@dataclass(slots=True)
class PlaceholderLLMClient:
    model: str = ""

    def summarize_paper(
        self,
        *,
        title: str,
        prompt: str,
        known_keywords: list[str],
        known_categories: list[str],
        pdf_attachment: PdfAttachment,
    ) -> SummaryResult:
        del prompt, pdf_attachment
        keywords = known_keywords[:3] or ["TODO-keyword"]
        categories = known_categories[:2] or ["TODO-category"]
        summary = {
            "notice": "This is a placeholder summary. Replace PlaceholderLLMClient.",
            "title": title,
            "research_question": "TODO",
            "method": "TODO",
            "experiments": "TODO",
            "findings": "TODO",
            "limitations": "TODO",
        }
        return SummaryResult(
            title=title,
            summary=json.dumps(summary, ensure_ascii=False, indent=2),
            keywords=keywords,
            categories=categories,
        )

    def draft_deep_note(
        self,
        *,
        title: str,
        prompt: str,
        related_note_paths: list[str],
        pdf_attachment: PdfAttachment,
    ) -> str:
        del prompt, pdf_attachment
        related = "\n".join(f"- {path}" for path in related_note_paths) or "- None"
        return (
            f"# {title}\n\n"
            "> This is a placeholder deep-analysis draft. Replace PlaceholderLLMClient.\n\n"
            "## 一句话贡献\n\nTODO\n\n"
            "## 背景与问题\n\nTODO\n\n"
            "## 方法拆解\n\nTODO\n\n"
            "## 实验与证据\n\nTODO\n\n"
            "## 与已有论文的关系\n\n"
            f"{related}\n\n"
            "## 可复现细节\n\nTODO\n\n"
            "## 值得追问的问题\n\nTODO\n\n"
            "## 个人研究启发\n\nTODO\n"
        )

    def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
        return (
            f"{current_note.rstrip()}\n\n"
            f"## Revision Request For {title}\n\n"
            f"{user_message.strip()}\n"
        )

    def answer_note_question(
        self,
        *,
        title: str,
        current_note: str,
        user_question: str,
    ) -> str:
        del current_note
        return f"[Placeholder answer for {title}] {user_question.strip()}"


@dataclass(slots=True)
class DeepSeekLLMClient:
    api_key: str
    model: str
    base_url: str = ""

    def summarize_paper(
        self,
        *,
        title: str,
        prompt: str,
        known_keywords: list[str],
        known_categories: list[str],
        pdf_attachment: PdfAttachment,
    ) -> SummaryResult:
        del known_keywords, known_categories
        response_text = self._chat(
            system_message="你是一个严谨的学术论文阅读助手。",
            user_message=self._build_pdf_user_message(
                text=(
                    f"论文标题：{title}\n\n"
                    f"{prompt}\n\n"
                    "请直接输出 JSON 对象，字段必须包含："
                    "summary, keywords, categories, conference, publication_time。"
                ),
                pdf_attachment=pdf_attachment,
            ),
        )
        payload = self._parse_json_object(response_text)
        return SummaryResult(
            title=title,
            summary=str(payload.get("summary") or response_text).strip(),
            keywords=self._normalize_string_list(payload.get("keywords"), fallback=["未分类关键词"]),
            categories=self._normalize_string_list(
                payload.get("categories"),
                fallback=["未分类"],
            ),
            conference=self._normalize_optional_string(payload.get("conference")),
            publication_time=self._normalize_optional_string(payload.get("publication_time")),
        )

    def draft_deep_note(
        self,
        *,
        title: str,
        prompt: str,
        related_note_paths: list[str],
        pdf_attachment: PdfAttachment,
    ) -> str:
        del related_note_paths
        return self._chat(
            system_message="你是一个严谨的学术研究助理。",
            user_message=self._build_pdf_user_message(
                text=f"论文标题：{title}\n\n{prompt}",
                pdf_attachment=pdf_attachment,
            ),
        ).strip()

    def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
        return self._chat(
            system_message="你是一个严谨的学术研究助理，请根据用户反馈修改论文笔记。",
            user_message=(
                f"论文标题：{title}\n\n"
                f"当前笔记：\n{current_note}\n\n"
                f"用户修改意见：\n{user_message}\n\n"
                "请返回完整修订后的笔记，不要解释。"
            ),
        ).strip()

    def answer_note_question(
        self,
        *,
        title: str,
        current_note: str,
        user_question: str,
    ) -> str:
        return self._chat(
            system_message="你是一个严谨的学术研究助理，请基于当前论文笔记回答用户问题。",
            user_message=(
                f"论文标题：{title}\n\n"
                f"当前笔记：\n{current_note}\n\n"
                f"用户问题：\n{user_question}\n\n"
                "请直接回答问题；如果当前笔记无法支持明确结论，要明确说明不确定点。"
            ),
        ).strip()

    def _chat(self, *, system_message: str, user_message: str | list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise ValueError("SHOLAR_API_KEY is required when SHOLAR_LLM_PROVIDER is not placeholder")
        if not self.base_url:
            raise ValueError("SHOLAR_BASE_URL is required when SHOLAR_LLM_PROVIDER is not placeholder")

        body = {
            "model": self.model or "deepseek-chat",
            "instructions": system_message,
            "input": [
                {
                    "role": "user",
                    "content": self._normalize_input_content(user_message),
                }
            ],
            "text": {"format": {"type": "text"}},
            "store": False,
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            url=self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
                text = raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"LLM request failed: {exc.reason}") from exc

        try:
            payload = json.loads(text)
        except JSONDecodeError as exc:
            preview = text[:300].replace("\n", " ").strip()
            raise ValueError(
                f"LLM response was not valid JSON. Content-Type: {content_type or 'unknown'}. "
                f"Body preview: {preview}"
            ) from exc

        if isinstance(payload.get("error"), dict):
            detail = json.dumps(payload["error"], ensure_ascii=False)
            raise ValueError(f"LLM response contained an error: {detail}")

        response_text = self._extract_response_text(payload)
        if response_text:
            return response_text
        raise ValueError("Responses API response content was empty")

    @staticmethod
    def _normalize_input_content(user_message: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(user_message, str):
            return [
                {
                    "type": "input_text",
                    "text": user_message,
                }
            ]
        return user_message

    @staticmethod
    def _extract_response_text(payload: dict[str, Any]) -> str:
        output = payload.get("output")
        if not isinstance(output, list):
            return ""

        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "output_text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
        return "\n".join(text_parts).strip()

    @staticmethod
    def _build_pdf_user_message(
        *,
        text: str,
        pdf_attachment: PdfAttachment,
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "input_text",
                "text": text,
            },
            {
                "type": "input_file",
                "filename": pdf_attachment.filename,
                "file_data": f"data:application/pdf;base64,{pdf_attachment.base64_data}",
            },
        ]

    @staticmethod
    def _parse_json_object(value: str) -> dict[str, Any]:
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _normalize_string_list(value: Any, *, fallback: list[str]) -> list[str]:
        if not isinstance(value, list):
            return fallback
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or fallback

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def build_llm_client(provider: str, model: str) -> LLMClient:
    if provider == "placeholder":
        return PlaceholderLLMClient(model=model)
    settings = get_settings()
    api_key = settings.api_key or os.getenv("SHOLAR_API_KEY", "")
    return DeepSeekLLMClient(
        api_key=api_key,
        model=model ,
        base_url=settings.base_url,
    )


class PaperRepository:
    """SQLite persistence used as the agent's long-term local memory."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    title TEXT PRIMARY KEY,
                    conference TEXT,
                    publication_time TEXT,
                    summary_generated INTEGER NOT NULL DEFAULT 0,
                    user_read INTEGER NOT NULL DEFAULT 0,
                    deep_analyzed INTEGER NOT NULL DEFAULT 0,
                    note_path TEXT,
                    pdf_path TEXT NOT NULL,
                    summary TEXT,
                    keywords_json TEXT NOT NULL DEFAULT '[]',
                    categories_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deep_analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_title TEXT NOT NULL,
                    final_note_path TEXT,
                    confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (paper_title) REFERENCES papers(title)
                )
                """
            )
            self._migrate_deep_analysis_sessions(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_papers_summary_user_read
                ON papers(summary_generated, user_read)
                """
            )
            conn.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def _migrate_deep_analysis_sessions(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(deep_analysis_sessions)").fetchall()
        }
        if "draft_note" not in columns:
            return

        conn.execute("ALTER TABLE deep_analysis_sessions RENAME TO deep_analysis_sessions_old")
        conn.execute(
            """
            CREATE TABLE deep_analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_title TEXT NOT NULL,
                final_note_path TEXT,
                confirmed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (paper_title) REFERENCES papers(title)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO deep_analysis_sessions (
                id,
                paper_title,
                final_note_path,
                confirmed,
                created_at,
                updated_at
            )
            SELECT
                id,
                paper_title,
                final_note_path,
                confirmed,
                created_at,
                updated_at
            FROM deep_analysis_sessions_old
            """
        )
        conn.execute("DROP TABLE deep_analysis_sessions_old")

    def upsert_scanned_paper(
        self,
        *,
        title: str,
        pdf_path: Path | str,
        conference: str | None = None,
        publication_time: str | None = None,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                    title,
                    conference,
                    publication_time,
                    pdf_path,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(title) DO UPDATE SET
                    pdf_path = excluded.pdf_path,
                    conference = COALESCE(papers.conference, excluded.conference),
                    publication_time = COALESCE(
                        papers.publication_time,
                        excluded.publication_time
                    ),
                    updated_at = excluded.updated_at
                """,
                (title, conference, publication_time, str(pdf_path), now, now),
            )

    def get_paper(self, title: str) -> Paper | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE title = ?", (title,)).fetchone()
        return row_to_paper(row) if row else None

    def list_all(self) -> list[Paper]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY updated_at DESC").fetchall()
        return [row_to_paper(row) for row in rows]

    def list_pending_summary(self, limit: int = 20) -> list[Paper]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE summary_generated = 0
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [row_to_paper(row) for row in rows]

    def list_unread_summaries(self, limit: int = 50) -> list[Paper]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE summary_generated = 1
                  AND user_read = 0
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [row_to_paper(row) for row in rows]

    def list_by_categories(
        self,
        categories: list[str],
        exclude_title: str | None = None,
    ) -> list[Paper]:
        if not categories:
            return []

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE note_path IS NOT NULL
                  AND deep_analyzed = 1
                ORDER BY updated_at DESC
                """
            ).fetchall()

        papers = [row_to_paper(row) for row in rows]
        category_set = set(categories)
        return [
            paper
            for paper in papers
            if paper.title != exclude_title and category_set.intersection(paper.categories)
        ]

    def save_summary(self, result: SummaryResult) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE papers
                SET conference = COALESCE(?, conference),
                    publication_time = COALESCE(?, publication_time),
                    summary_generated = 1,
                    summary = ?,
                    keywords_json = ?,
                    categories_json = ?,
                    updated_at = ?
                WHERE title = ?
                """,
                (
                    result.conference,
                    result.publication_time,
                    result.summary,
                    json.dumps(result.keywords, ensure_ascii=False),
                    json.dumps(result.categories, ensure_ascii=False),
                    now,
                    result.title,
                ),
            )

    def mark_user_read(self, title: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE papers
                SET user_read = 1,
                    updated_at = ?
                WHERE title = ?
                """,
                (now, title),
            )

    def mark_deep_analyzed(self, *, title: str, note_path: Path | str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE papers
                SET user_read = 1,
                    deep_analyzed = 1,
                    note_path = ?,
                    updated_at = ?
                WHERE title = ?
                """,
                (str(note_path), now, title),
            )

    def create_deep_analysis_session(self, *, paper_title: str) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO deep_analysis_sessions (
                    paper_title,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?)
                """,
                (paper_title, now, now),
            )
            return int(cursor.lastrowid)

    def confirm_deep_analysis_session(self, *, session_id: int, note_path: Path | str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE deep_analysis_sessions
                SET final_note_path = ?,
                    confirmed = 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(note_path), now, session_id),
            )


def iter_pdf_paths(root: Path | str) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*.pdf") if path.is_file())


def extract_pdf_title(path: Path | str) -> str:
    pdf_path = Path(path)
    try:
        reader = PdfReader(str(pdf_path))
        metadata_title = reader.metadata.title if reader.metadata else None
        if metadata_title:
            return normalize_title(metadata_title)

        first_page_text = reader.pages[0].extract_text() if reader.pages else ""
        first_line = first_non_empty_line(first_page_text)
        if first_line:
            return normalize_title(first_line)
    except Exception:
        pass

    return normalize_title(pdf_path.stem)


def encode_pdf_base64(path: Path | str) -> str:
    pdf_path = Path(path)
    return base64.b64encode(pdf_path.read_bytes()).decode("ascii")


def build_pdf_attachment(path: Path | str) -> PdfAttachment:
    pdf_path = Path(path)
    return PdfAttachment(
        filename=pdf_path.name,
        base64_data=encode_pdf_base64(pdf_path),
    )


def get_repository() -> PaperRepository:
    settings = get_settings()
    return PaperRepository(settings.db_path)


def get_llm() -> LLMClient:
    settings = get_settings()
    return build_llm_client(settings.llm_provider, settings.llm_model)


def default_pdf_root() -> Path:
    return get_settings().pdf_root


def collect_known_terms(repository: PaperRepository) -> tuple[list[str], list[str]]:
    keywords: set[str] = set()
    categories: set[str] = set()
    for paper in repository.list_all():
        keywords.update(paper.keywords)
        categories.update(paper.categories)
    return sorted(keywords), sorted(categories)


def serialize_paper(paper: Paper) -> dict:
    return {
        "title": paper.title,
        "conference": paper.conference,
        "publication_time": paper.publication_time,
        "summary_generated": paper.summary_generated,
        "user_read": paper.user_read,
        "deep_analyzed": paper.deep_analyzed,
        "note_path": paper.note_path,
        "pdf_path": paper.pdf_path,
        "summary": paper.summary,
        "keywords": paper.keywords,
        "categories": paper.categories,
    }


def read_related_notes(paths: list[str]) -> str:
    chunks: list[str] = []
    for path in paths:
        note_path = Path(path)
        if not note_path.exists():
            continue
        chunks.append(f"\n\n--- {note_path} ---\n{note_path.read_text(encoding='utf-8')}")
    return "\n".join(chunks).strip()


def read_note(path: Path | str) -> str:
    note_path = Path(path)
    return note_path.read_text(encoding="utf-8")


def note_output_path_for_paper(paper: Paper) -> Path:
    pdf_path = Path(paper.pdf_path)
    return pdf_path.with_name(f"{safe_filename(paper.title)}.notes.md")


def write_draft_note(*, paper: Paper, draft_note: str) -> Path:
    note_path = note_output_path_for_paper(paper)
    note_path.write_text(draft_note, encoding="utf-8")
    return note_path


def write_final_note(*, paper: Paper, final_note: str) -> Path:
    note_path = note_output_path_for_paper(paper)
    note_path.write_text(final_note, encoding="utf-8")
    return note_path


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE)
    return cleaned.strip("._") or "paper"


def row_to_paper(row: sqlite3.Row | dict[str, Any]) -> Paper:
    return Paper(
        title=row["title"],
        conference=row["conference"],
        publication_time=row["publication_time"],
        summary_generated=bool(row["summary_generated"]),
        user_read=bool(row["user_read"]),
        deep_analyzed=bool(row["deep_analyzed"]),
        note_path=row["note_path"],
        pdf_path=row["pdf_path"],
        summary=row["summary"],
        keywords=json.loads(row["keywords_json"] or "[]"),
        categories=json.loads(row["categories_json"] or "[]"),
        created_at=parse_dt(row["created_at"]),
        updated_at=parse_dt(row["updated_at"]),
    )


def parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def first_non_empty_line(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return None


def normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    return title.strip(" ._-")
