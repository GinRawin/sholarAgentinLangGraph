from __future__ import annotations

import json
import re
import sqlite3
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


SCHEMA_VERSION = 1

SUMMARY_TEMPLATE = """你是一个严谨的学术论文阅读助手。

请阅读论文原文，生成结构化摘要，并尽量从已有关键词和分类中复用合适条目。

已有关键词：
{known_keywords}

已有分类：
{known_categories}

输出要求：
1. 论文标题
2. 论文所属会议，如果无法判断则留空
3. 论文发表时间，如果无法判断则留空
4. 研究问题
5. 核心方法
6. 实验设置
7. 主要结论
8. 局限性
9. 关键词列表
10. 分类列表

论文原文：
{paper_text}
"""


DEEP_ANALYSIS_TEMPLATE = """你是一个学术研究助理，需要生成可迭代修改的论文笔记初稿。

请阅读新论文原文，并参考同分类历史论文笔记。历史笔记只作为比较和关联材料，不要编造原文中没有的信息。

同分类历史笔记：
{related_notes}

输出笔记结构：
1. 一句话贡献
2. 背景与问题
3. 方法拆解
4. 实验与证据
5. 与已有论文的关系
6. 可复现细节
7. 值得追问的问题
8. 个人研究启发

新论文原文：
{paper_text}
"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="scholar_")

    db_path: Path = Field(default=Path("./data/scholar_agent.sqlite3"))
    pdf_root: Path = Field(default=Path("./papers"))
    llm_provider: str = Field(default="placeholder")
    llm_model: str = Field(default="")


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
    ) -> SummaryResult:
        ...

    def draft_deep_note(self, *, title: str, prompt: str, related_note_paths: list[str]) -> str:
        ...

    def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
        ...


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
    ) -> SummaryResult:
        del prompt
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

    def draft_deep_note(self, *, title: str, prompt: str, related_note_paths: list[str]) -> str:
        del prompt
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


def build_llm_client(provider: str, model: str) -> LLMClient:
    if provider == "placeholder":
        return PlaceholderLLMClient(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")


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
                    draft_note TEXT,
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

def create_deep_analysis_session(self, *, paper_title: str, draft_note: str) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO deep_analysis_sessions (
                    paper_title,
                    draft_note,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (paper_title, draft_note, now, now),
            )
            return int(cursor.lastrowid)


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


def extract_pdf_text(path: Path | str, *, max_pages: int | None = None) -> str:
    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(f"\n\n--- Page {page_number} ---\n{text.strip()}")
    return "\n".join(chunks).strip()


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


def write_final_note(*, paper: Paper, final_note: str) -> Path:
    pdf_path = Path(paper.pdf_path)
    note_path = pdf_path.with_name(f"{safe_filename(paper.title)}.notes.md")
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
