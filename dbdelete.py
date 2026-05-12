from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "scholar_agent.sqlite3"


@dataclass(slots=True)
class PaperDeletionPlan:
    title: str
    pdf_path: Path | None
    note_path: Path | None
    deep_analysis_session_count: int


@dataclass(slots=True)
class PaperDeletionResult:
    title: str
    deleted_files: list[Path]
    missing_files: list[Path]
    deleted_db_rows: dict[str, int]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def load_env_value(key: str) -> str | None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, value = stripped.split("=", 1)
        if env_key.strip() == key:
            return value.strip()
    return None


def load_db_path() -> Path:
    env_value = load_env_value("SHOLAR_DB_PATH")
    if env_value:
        return resolve_project_path(env_value)
    return resolve_project_path(os.getenv("SHOLAR_DB_PATH", str(DEFAULT_DB_PATH)))


def resolve_stored_path(value: str | None) -> Path | None:
    if not value:
        return None
    return resolve_project_path(value)


def build_deletion_plan(conn: sqlite3.Connection, title: str) -> PaperDeletionPlan | None:
    row = conn.execute(
        """
        SELECT title, pdf_path, note_path
        FROM papers
        WHERE title = ?
        """,
        (title,),
    ).fetchone()
    if row is None:
        return None

    deep_analysis_session_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM deep_analysis_sessions
            WHERE paper_title = ?
            """,
            (title,),
        ).fetchone()[0]
    )
    return PaperDeletionPlan(
        title=str(row["title"]),
        pdf_path=resolve_stored_path(row["pdf_path"]),
        note_path=resolve_stored_path(row["note_path"]),
        deep_analysis_session_count=deep_analysis_session_count,
    )


def find_similar_titles(conn: sqlite3.Connection, title: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT title
        FROM papers
        WHERE title LIKE ?
        ORDER BY updated_at DESC
        LIMIT 10
        """,
        (f"%{title}%",),
    ).fetchall()
    return [str(row["title"]) for row in rows]


def delete_paper_bundle(*, db_path: Path | str, title: str) -> PaperDeletionResult:
    resolved_db_path = resolve_project_path(db_path)
    if not resolved_db_path.exists():
        raise FileNotFoundError(f"Database file does not exist: {resolved_db_path}")

    conn = sqlite3.connect(resolved_db_path)
    conn.row_factory = sqlite3.Row
    try:
        plan = build_deletion_plan(conn, title)
        if plan is None:
            similar_titles = find_similar_titles(conn, title)
            if similar_titles:
                suggestion_text = "\n".join(f"- {item}" for item in similar_titles)
                raise ValueError(f"Paper not found: {title}\nPossible matches:\n{suggestion_text}")
            raise ValueError(f"Paper not found: {title}")
    finally:
        conn.close()

    deleted_files: list[Path] = []
    missing_files: list[Path] = []
    file_errors: list[str] = []

    file_paths: list[Path] = []
    if plan.note_path is not None:
        file_paths.append(plan.note_path)
    if plan.pdf_path is not None:
        file_paths.append(plan.pdf_path)
    seen_paths: set[Path] = set()
    unique_paths: list[Path] = []
    for path in file_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_paths.append(path)

    for path in unique_paths:
        if not path.exists():
            missing_files.append(path)
            continue
        if path.is_dir():
            file_errors.append(f"Expected file but found directory: {path}")
            continue
        try:
            path.unlink()
            deleted_files.append(path)
        except OSError as exc:
            file_errors.append(f"Failed to delete {path}: {exc}")

    if file_errors:
        error_text = "\n".join(file_errors)
        raise RuntimeError(f"Aborting database deletion because file cleanup failed:\n{error_text}")

    conn = sqlite3.connect(resolved_db_path)
    try:
        with conn:
            sessions_deleted = conn.execute(
                "DELETE FROM deep_analysis_sessions WHERE paper_title = ?",
                (title,),
            ).rowcount
            papers_deleted = conn.execute(
                "DELETE FROM papers WHERE title = ?",
                (title,),
            ).rowcount
    finally:
        conn.close()

    return PaperDeletionResult(
        title=title,
        deleted_files=deleted_files,
        missing_files=missing_files,
        deleted_db_rows={
            "papers": int(papers_deleted or 0),
            "deep_analysis_sessions": int(sessions_deleted or 0),
        },
    )


def confirm_plan(plan: PaperDeletionPlan, *, db_path: Path) -> bool:
    print(f"Database: {db_path}")
    print(f"Paper: {plan.title}")
    print(f"PDF: {plan.pdf_path or 'N/A'}")
    print(f"Note: {plan.note_path or 'N/A'}")
    print(f"Deep analysis sessions: {plan.deep_analysis_session_count}")
    answer = input("Delete this paper and all related files? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete one paper and all related local data from the scholar agent.",
    )
    parser.add_argument("title", help="Exact paper title stored in the database.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without interactive confirmation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = load_db_path()
    if not db_path.exists():
        print(f"Database file does not exist: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        plan = build_deletion_plan(conn, args.title)
        if plan is None:
            similar_titles = find_similar_titles(conn, args.title)
            print(f"Paper not found: {args.title}", file=sys.stderr)
            if similar_titles:
                print("Possible matches:", file=sys.stderr)
                for title in similar_titles:
                    print(f"- {title}", file=sys.stderr)
            return 1
    finally:
        conn.close()

    if not args.yes and not confirm_plan(plan, db_path=db_path):
        print("Cancelled.")
        return 0

    try:
        result = delete_paper_bundle(db_path=db_path, title=args.title)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Deleted paper: {result.title}")
    for table_name, count in result.deleted_db_rows.items():
        print(f"- {table_name}: {count}")
    print(f"- deleted files: {len(result.deleted_files)}")
    print(f"- missing files: {len(result.missing_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
