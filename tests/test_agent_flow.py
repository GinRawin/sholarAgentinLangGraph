from __future__ import annotations

from pathlib import Path

from scholar_agent import agent
from scholar_agent.utils import nodes, tools


def test_route_after_select_unsummarized():
    assert agent.route_after_select_unsummarized({"status": "unsummarized_paper_selected"}) == "generate_summary"
    assert agent.route_after_select_unsummarized({"status": "no_unsummarized_paper"}) == "load_review_queue"


def test_route_after_review_queue():
    assert agent.route_after_review_queue({"review_queue": [{"title": "paper"}]}) == "human_summary_review"
    assert agent.route_after_review_queue({"review_queue": []}) == "end"


def test_route_after_summary_decision():
    assert agent.route_after_summary_decision({"user_decision": "deep_analysis"}) == "prepare_deep_analysis_context"
    assert agent.route_after_summary_decision({"user_decision": "skip"}) == "load_review_queue"


def test_route_after_note_review():
    assert agent.route_after_note_review({"note_review_action": "confirm"}) == "save_final_note"
    assert agent.route_after_note_review({"note_review_action": "finish_qa"}) == "revise_deep_analysis_note"
    assert agent.route_after_note_review({"note_review_action": "ask_question"}) == "question_and_answer"


def test_route_after_question_and_answer():
    assert agent.route_after_question_and_answer({}) == "human_note_review"


def test_human_summary_review_normalizes_numeric_decision(monkeypatch):
    monkeypatch.setattr(nodes, "interrupt", lambda payload: 0)

    result = nodes.human_summary_review_node(
        {
            "review_queue": [
                {
                    "title": "paper",
                    "summary": "summary",
                    "pdf_path": "papers/paper.pdf",
                    "keywords": [],
                    "categories": [],
                }
            ]
        }
    )

    assert result["status"] == "summary_reviewed_by_user"
    assert result["title"] == "paper"
    assert result["user_decision"] == "deep_analysis"


def test_human_note_review_reads_note_from_disk(monkeypatch, tmp_path):
    note_path = tmp_path / "paper.notes.md"
    note_path.write_text("# Draft From Disk\n\ncontent", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_interrupt(payload: object) -> object:
        captured["payload"] = payload
        return "confirm"

    monkeypatch.setattr(nodes, "interrupt", fake_interrupt)

    result = nodes.human_note_review_node(
        {
            "title": "paper",
            "note_path": str(note_path),
        }
    )

    assert result["status"] == "note_confirmed_by_user"
    assert result["note_review_action"] == "confirm"
    assert "# Draft From Disk" in str(captured["payload"])


def test_human_note_review_routes_done_to_finish_qa(monkeypatch):
    monkeypatch.setattr(nodes, "interrupt", lambda payload: "done")

    result = nodes.human_note_review_node(
        {
            "title": "paper",
            "draft_note": "# Draft",
            "qa_history": [{"question": "Q1", "answer": "A1"}],
            "latest_answer": "A1",
        }
    )

    assert result["status"] == "note_questions_completed_by_user"
    assert result["note_review_action"] == "finish_qa"


def test_question_and_answer_node_records_history_and_returns_answer(monkeypatch, tmp_path):
    calls: dict[str, str] = {}

    class FakeLLM:
        def answer_note_question(
            self,
            *,
            title: str,
            current_note: str,
            user_question: str,
            pdf_attachment: tools.PdfAttachment,
        ) -> str:
            calls["title"] = title
            calls["current_note"] = current_note
            calls["user_question"] = user_question
            calls["pdf_filename"] = pdf_attachment.filename
            return "这是问题的答案"

    class FakeRepo:
        def get_paper(self, title: str) -> tools.Paper:
            return tools.Paper(title=title, pdf_path=str(tmp_path / "paper.pdf"))

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(nodes, "get_repository", lambda: FakeRepo())

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 test")
    note_path = tmp_path / "paper.notes.md"
    note_path.write_text("# Draft\n\nexisting", encoding="utf-8")

    result = nodes.question_and_answer_node(
        {
            "title": "paper",
            "note_path": str(note_path),
            "latest_question": "这个方法为什么有效？",
            "qa_history": [{"question": "Q0", "answer": "A0"}],
        }
    )

    assert result["status"] == "question_answered"
    assert result["latest_answer"] == "这是问题的答案"
    assert result["question_answer"] == "这是问题的答案"
    assert result["qa_history"] == [
        {"question": "Q0", "answer": "A0"},
        {"question": "这个方法为什么有效？", "answer": "这是问题的答案"},
    ]
    assert calls["title"] == "paper"
    assert calls["current_note"] == "# Draft\n\nexisting"
    assert calls["user_question"] == "这个方法为什么有效？"
    assert calls["pdf_filename"] == "paper.pdf"


def test_draft_node_writes_note_file(monkeypatch, tmp_path):
    class FakeLLM:
        def draft_deep_note(
            self,
            *,
            title: str,
            prompt: str,
            related_note_paths: list[str],
            pdf_attachment: tools.PdfAttachment,
        ) -> str:
            del title, prompt, related_note_paths
            assert pdf_attachment.filename == "paper.pdf"
            assert pdf_attachment.base64_data
            return "# Draft\n\ncontent"

    class FakeRepo:
        def get_paper(self, title: str) -> tools.Paper:
            return tools.Paper(title=title, pdf_path=str(tmp_path / "paper.pdf"))

        def create_deep_analysis_session(self, *, paper_title: str) -> int:
            del paper_title
            return 7

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(nodes, "get_repository", lambda: FakeRepo())
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 test")

    result = nodes.draft_deep_analysis_note_node(
        {
            "title": "paper",
            "deep_analysis_prompt": "prompt",
            "related_note_paths": [],
        }
    )

    note_path = Path(result["note_path"])
    assert result["status"] == "deep_analysis_draft_created"
    assert result["deep_analysis_session_id"] == 7
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "# Draft\n\ncontent"


def test_revise_node_uses_qa_history_to_rewrite_note(monkeypatch, tmp_path):
    calls: dict[str, str] = {}

    class FakeLLM:
        def revise_note(
            self,
            *,
            title: str,
            current_note: str,
            user_message: str,
        ) -> str:
            del title
            calls["revision_input"] = current_note
            calls["revision_prompt"] = user_message
            return current_note + "\n\n## 更新\n\n已补充说明"

    class FakeRepo:
        def get_paper(self, title: str) -> tools.Paper:
            return tools.Paper(title=title, pdf_path=str(tmp_path / "paper.pdf"))

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(nodes, "get_repository", lambda: FakeRepo())

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 test")
    note_path = tmp_path / "paper.notes.md"
    note_path.write_text("# Draft\n\nexisting", encoding="utf-8")

    result = nodes.revise_deep_analysis_note_node(
        {
            "title": "paper",
            "note_path": str(note_path),
            "qa_history": [
                {
                    "question": "这个方法为什么有效？",
                    "answer": "因为它利用了更稳定的特征表示。",
                }
            ],
        }
    )

    assert result["status"] == "deep_analysis_draft_revised"
    assert "这个方法为什么有效？" in calls["revision_prompt"]
    assert "因为它利用了更稳定的特征表示。" in calls["revision_prompt"]
    assert "已补充说明" in note_path.read_text(encoding="utf-8")


def test_save_final_note_marks_paper_and_confirms_session(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class FakeRepo:
        def get_paper(self, title: str) -> tools.Paper:
            return tools.Paper(title=title, pdf_path=str(tmp_path / "paper.pdf"))

        def mark_deep_analyzed(self, *, title: str, note_path: Path | str) -> None:
            calls["mark_deep_analyzed"] = (title, str(note_path))

        def confirm_deep_analysis_session(self, *, session_id: int, note_path: Path | str) -> None:
            calls["confirm_deep_analysis_session"] = (session_id, str(note_path))

    monkeypatch.setattr(nodes, "get_repository", lambda: FakeRepo())
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 test")

    result = nodes.save_final_note_node(
        {
            "title": "paper",
            "final_note": "# Final\n\ncontent",
            "deep_analysis_session_id": 12,
        }
    )

    assert result["status"] == "final_note_saved"
    saved_path = Path(result["note_path"])
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "# Final\n\ncontent"
    assert calls["mark_deep_analyzed"] == ("paper", str(saved_path))
    assert calls["confirm_deep_analysis_session"] == (12, str(saved_path))


def test_prepare_deep_analysis_context_sorts_related_notes(monkeypatch, tmp_path):
    target = tools.Paper(
        title="target",
        pdf_path=str(tmp_path / "target.pdf"),
        keywords=["agent", "memory"],
        categories=["survey"],
    )
    less_related_note = tmp_path / "less.md"
    more_related_note = tmp_path / "more.md"
    less_related_note.write_text("less", encoding="utf-8")
    more_related_note.write_text("more", encoding="utf-8")

    class FakeRepo:
        def get_paper(self, title: str) -> tools.Paper | None:
            return target if title == "target" else None

        def list_by_categories(self, categories: list[str], exclude_title: str | None = None) -> list[tools.Paper]:
            del categories, exclude_title
            return [
                tools.Paper(
                    title="less-related",
                    pdf_path="less.pdf",
                    keywords=["agent"],
                    categories=["survey"],
                    note_path=str(less_related_note),
                ),
                tools.Paper(
                    title="more-related",
                    pdf_path="more.pdf",
                    keywords=["agent", "memory"],
                    categories=["survey"],
                    note_path=str(more_related_note),
                ),
            ]

        def mark_user_read(self, title: str) -> None:
            assert title == "target"

    monkeypatch.setattr(nodes, "get_repository", lambda: FakeRepo())

    result = nodes.prepare_deep_analysis_context_node({"title": "target"})

    assert result["status"] == "deep_analysis_context_ready"
    assert result["related_note_paths"] == [str(more_related_note), str(less_related_note)]
    assert "more" in result["related_notes"]
    assert "less" in result["related_notes"]
    assert "同分类历史笔记" in result["deep_analysis_prompt"]
