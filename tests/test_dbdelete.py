from __future__ import annotations

import sqlite3
from pathlib import Path

import dbdelete
from scholar_agent.utils import tools


def test_delete_paper_bundle_removes_db_rows_and_files(tmp_path):
    db_path = tmp_path / "scholar_agent.sqlite3"
    pdf_path = tmp_path / "papers" / "paper.pdf"
    note_path = tmp_path / "papers" / "paper.notes.md"
    asset_dir = tmp_path / "assets" / "paper"
    asset_path = asset_dir / "page_001_image_001.png"

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")
    note_path.write_text("# Note", encoding="utf-8")
    asset_path.write_bytes(b"png")

    repo = tools.PaperRepository(db_path)
    repo.init_db()
    repo.upsert_scanned_paper(title="paper", pdf_path=pdf_path)
    repo.mark_deep_analyzed(title="paper", note_path=note_path)
    session_id = repo.create_deep_analysis_session(paper_title="paper")
    repo.confirm_deep_analysis_session(session_id=session_id, note_path=note_path)
    repo.replace_graph_table_assets(
        "paper",
        [
            tools.GraphTableAsset(
                paper_title="paper",
                asset_index=1,
                page_number=1,
                asset_type="image",
                source_name="Image1",
                asset_path=str(asset_path),
                surrounding_text="page text",
            )
        ],
    )

    result = dbdelete.delete_paper_bundle(db_path=db_path, title="paper")

    assert result.deleted_db_rows == {
        "papers": 1,
        "deep_analysis_sessions": 1,
        "graph_table_assets": 1,
    }
    assert not pdf_path.exists()
    assert not note_path.exists()
    assert not asset_path.exists()
    assert not asset_dir.exists()

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM deep_analysis_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM graph_table_assets").fetchone()[0] == 0
    finally:
        conn.close()


def test_delete_paper_bundle_raises_for_missing_title(tmp_path):
    db_path = tmp_path / "scholar_agent.sqlite3"
    repo = tools.PaperRepository(db_path)
    repo.init_db()

    try:
        dbdelete.delete_paper_bundle(db_path=db_path, title="missing")
    except ValueError as exc:
        assert "Paper not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing title")
