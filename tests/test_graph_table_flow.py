from __future__ import annotations

from pathlib import Path

from PIL import Image

from scholar_agent import agent
from scholar_agent.utils import nodes, tools


def test_parse_graph_table_tool_call():
    payload = tools.parse_graph_table_tool_call(
        '{"tool":"getGraphorTableDetails","asset_index":2,"question":"图里最高的数值是多少？"}'
    )
    assert payload == {
        "tool": "getGraphorTableDetails",
        "asset_index": 2,
        "question": "图里最高的数值是多少？",
    }


def test_parse_graph_table_tool_call_with_fenced_json():
    payload = tools.parse_graph_table_tool_call(
        '```json\n'
        '{"tool":"getGraphorTableDetails","asset_index":2,"question":"图里最高的数值是多少？"}\n'
        '```'
    )
    assert payload == {
        "tool": "getGraphorTableDetails",
        "asset_index": 2,
        "question": "图里最高的数值是多少？",
    }


def test_parse_graph_table_tool_call_with_wrapped_text():
    payload = tools.parse_graph_table_tool_call(
        '我需要查询图表细节：{"tool":"getGraphorTableDetails","asset_index":2,"question":"图里最高的数值是多少？"}'
    )
    assert payload == {
        "tool": "getGraphorTableDetails",
        "asset_index": 2,
        "question": "图里最高的数值是多少？",
    }


def test_extract_graph_table_assets_from_sample_pdf():
    pdf_path = Path("papers/Kim 等 - 2026 - The Attack and Defense Landscape of Agentic AI A Comprehensive Survey.pdf")
    assets = tools.extract_graph_table_assets(
        paper_title="sample-paper",
        pdf_path=pdf_path,
    )
    assert assets
    assert assets[0].page_number >= 1
    assert Path(assets[0].asset_path).exists()


def test_draft_node_requests_graph_table_tool(monkeypatch):
    class FakeLLM:
        def draft_deep_note(
            self,
            *,
            title: str,
            prompt: str,
            related_note_paths: list[str],
            pdf_attachment: tools.PdfAttachment,
        ) -> str:
            del title, prompt, related_note_paths, pdf_attachment
            return (
                '{"tool":"getGraphorTableDetails","asset_index":1,'
                '"question":"表格里的最佳结果是多少？"}'
            )

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        nodes,
        "get_repository",
        lambda: type("Repo", (), {"get_paper": lambda self, title: tools.Paper(title=title, pdf_path="papers/sample.pdf")})(),
    )

    state = {
        "title": "paper",
        "deep_analysis_prompt": "prompt",
        "related_note_paths": [],
        "graph_table_index": [
            {"asset_index": 1, "page_number": 3, "summary": "这是一个实验结果表"}
        ],
        "graph_table_qa_notes": [],
    }
    result = nodes.draft_deep_analysis_note_node(state)
    assert result["status"] == "graph_table_tool_requested"
    assert result["graph_table_question_index"] == 1
    assert "最佳结果" in result["graph_table_question"]


def test_revise_node_requests_graph_table_tool(monkeypatch):
    class FakeLLM:
        def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
            del title, current_note, user_message
            return (
                '{"tool":"getGraphorTableDetails","asset_index":4,'
                '"question":"图4中的横轴和纵轴分别是什么？"}'
            )

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    note_path = Path("tests/tmp_revise_tool.md")
    note_path.write_text("draft", encoding="utf-8")

    state = {
        "title": "paper",
        "note_path": str(note_path),
        "user_message": "请补充图表分析",
        "graph_table_qa_notes": [],
    }
    result = nodes.revise_deep_analysis_note_node(state)
    note_path.unlink()
    assert result["status"] == "graph_table_tool_requested"
    assert result["graph_table_question_index"] == 4
    assert "横轴" in result["graph_table_question"]


def test_revise_node_includes_graph_table_index_in_prompt(monkeypatch):
    captured: dict[str, str] = {}

    class FakeLLM:
        def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
            del title, current_note
            captured["user_message"] = user_message
            return "revised"

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())

    state = {
        "title": "paper",
        "note_path": str(Path("tests/tmp_revise_prompt.md")),
        "user_message": "请核对表格细节",
        "graph_table_index": [
            {"asset_index": 4, "page_number": 8, "summary": "这是一个性能对比表"}
        ],
        "graph_table_qa_notes": [],
    }
    Path(state["note_path"]).write_text("draft", encoding="utf-8")
    result = nodes.revise_deep_analysis_note_node(state)
    Path(state["note_path"]).unlink()
    assert result["status"] == "deep_analysis_draft_revised"
    assert "可用图表索引" in captured["user_message"]
    assert "性能对比表" in captured["user_message"]
    assert "getGraphorTableDetails" in captured["user_message"]


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
            del title, prompt, related_note_paths, pdf_attachment
            return "# Draft\n\ncontent"

    class FakeRepo:
        def get_paper(self, title: str) -> tools.Paper:
            return tools.Paper(title=title, pdf_path=str(tmp_path / "paper.pdf"))

        def create_deep_analysis_session(self, *, paper_title: str) -> int:
            del paper_title
            return 7

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(nodes, "get_repository", lambda: FakeRepo())

    result = nodes.draft_deep_analysis_note_node(
        {
            "title": "paper",
            "deep_analysis_prompt": "prompt",
            "related_note_paths": [],
            "graph_table_index": [],
            "graph_table_qa_notes": [],
        }
    )

    note_path = Path(result["note_path"])
    assert result["status"] == "deep_analysis_draft_created"
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "# Draft\n\ncontent"


def test_revise_node_updates_note_file(monkeypatch, tmp_path):
    class FakeLLM:
        def revise_note(self, *, title: str, current_note: str, user_message: str) -> str:
            del title, user_message
            return current_note + "\n\nrevised"

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())

    note_path = tmp_path / "paper.notes.md"
    note_path.write_text("# Draft\n\ncontent", encoding="utf-8")

    result = nodes.revise_deep_analysis_note_node(
        {
            "title": "paper",
            "note_path": str(note_path),
            "user_message": "请补充细节",
            "graph_table_index": [],
            "graph_table_qa_notes": [],
        }
    )

    assert result["status"] == "deep_analysis_draft_revised"
    assert note_path.read_text(encoding="utf-8").endswith("revised")


def test_human_note_review_skips_interrupt_for_graph_table_tool_call(monkeypatch, tmp_path):
    note_path = tmp_path / "paper.notes.md"
    note_path.write_text(
        '{"tool":"getGraphorTableDetails","asset_index":3,"question":"表3中的最佳结果是多少？"}',
        encoding="utf-8",
    )

    def fail_interrupt(payload: object) -> object:
        del payload
        raise AssertionError("interrupt should not be called for graph table tool output")

    monkeypatch.setattr(nodes, "interrupt", fail_interrupt)

    result = nodes.human_note_review_node(
        {
            "title": "paper",
            "note_path": str(note_path),
            "user_message": "请补充表格细节",
        }
    )

    assert result["status"] == "graph_table_tool_requested"
    assert result["graph_table_question_index"] == 3
    assert "最佳结果" in result["graph_table_question"]
    assert result["graph_table_request_origin"] == "revise"


def test_get_graph_or_table_details_node(monkeypatch, tmp_path):
    class FakeVision:
        def answer_graph_table_question(
            self,
            *,
            title: str,
            page_hint: str,
            page_text: str,
            asset_path: Path,
            question: str,
        ) -> str:
            del title, page_hint, page_text, asset_path
            return f"answer: {question}"

    db_path = tmp_path / "test.sqlite3"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "sample.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    repo = tools.PaperRepository(db_path)
    repo.init_db()
    repo.upsert_scanned_paper(title="paper", pdf_path="papers/sample.pdf")
    repo.replace_graph_table_assets(
        "paper",
        [
            tools.GraphTableAsset(
                paper_title="paper",
                asset_index=1,
                page_number=2,
                asset_type="image",
                source_name="Image1",
                asset_path=str(image_path),
                surrounding_text="page text",
            )
        ],
    )

    monkeypatch.setattr(nodes, "get_repository", lambda: repo)
    monkeypatch.setattr(nodes, "get_vision_client", lambda: FakeVision())

    result = nodes.get_graph_or_table_details_node(
        {
            "title": "paper",
            "graph_table_question_index": 1,
            "graph_table_question": "这个图的主要趋势是什么？",
            "graph_table_qa_notes": [],
        }
    )
    assert result["status"] == "graph_table_question_answered"
    assert "主要趋势" in result["graph_table_question_result"]
    assert result["graph_table_qa_notes"]


def test_get_graph_table_content_prefetch_limit(monkeypatch, tmp_path):
    class FakeVision:
        def __init__(self) -> None:
            self.calls = 0

        def summarize_graph_table(
            self,
            *,
            title: str,
            page_hint: str,
            page_text: str,
            asset_path: Path,
        ) -> str:
            del title, page_hint, page_text, asset_path
            self.calls += 1
            return f"vision-summary-{self.calls}"

    db_path = tmp_path / "test.sqlite3"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    repo = tools.PaperRepository(db_path)
    repo.init_db()
    repo.upsert_scanned_paper(title="paper", pdf_path="papers/sample.pdf")

    assets: list[tools.GraphTableAsset] = []
    for index in range(25):
        image_path = asset_dir / f"sample_{index}.png"
        Image.new("RGB", (20, 20), color="white").save(image_path)
        assets.append(
            tools.GraphTableAsset(
                paper_title="paper",
                asset_index=index + 1,
                page_number=1,
                asset_type="image",
                source_name=f"Image{index + 1}",
                asset_path=str(image_path),
                surrounding_text="page text",
            )
        )
    repo.replace_graph_table_assets("paper", assets)

    fake_vision = FakeVision()
    monkeypatch.setattr(nodes, "get_repository", lambda: repo)
    monkeypatch.setattr(nodes, "get_vision_client", lambda: fake_vision)
    monkeypatch.setattr(nodes, "default_graph_table_prefetch_limit", lambda: 20)

    result = nodes.get_graph_table_content_node(
        {
            "title": "paper",
            "paper_text": "paper text",
            "graph_table_qa_notes": [],
        }
    )

    assert result["status"] == "graph_table_content_loaded"
    assert len(result["graph_table_index"]) == 25
    assert fake_vision.calls == 20
    assert result["graph_table_index"][0]["summary"] == "vision-summary-1"
    assert "图表 25 位于第 1 页" in str(result["graph_table_index"][-1]["summary"])


def test_get_graph_table_content_injects_summaries_into_paper_text_and_prompt(monkeypatch, tmp_path):
    class FakeVision:
        def summarize_graph_table(
            self,
            *,
            title: str,
            page_hint: str,
            page_text: str,
            asset_path: Path,
        ) -> str:
            del title, page_hint, page_text, asset_path
            return "这是图表摘要"

    db_path = tmp_path / "test.sqlite3"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "sample.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    repo = tools.PaperRepository(db_path)
    repo.init_db()
    repo.upsert_scanned_paper(title="paper", pdf_path="papers/sample.pdf")
    repo.replace_graph_table_assets(
        "paper",
        [
            tools.GraphTableAsset(
                paper_title="paper",
                asset_index=1,
                page_number=2,
                asset_type="image",
                source_name="Image1",
                asset_path=str(image_path),
                surrounding_text="page text",
            )
        ],
    )

    monkeypatch.setattr(nodes, "get_repository", lambda: repo)
    monkeypatch.setattr(nodes, "get_vision_client", lambda: FakeVision())
    monkeypatch.setattr(nodes, "default_graph_table_prefetch_limit", lambda: 20)

    result = nodes.get_graph_table_content_node(
        {
            "title": "paper",
            "paper_text": "原始正文",
            "related_notes": "历史笔记",
            "graph_table_qa_notes": [],
        }
    )

    assert result["status"] == "graph_table_content_loaded"
    assert "Extracted Graph/Table Summaries" in result["paper_text"]
    assert "这是图表摘要" in result["paper_text"]
    assert "这是图表摘要" in result["deep_analysis_prompt"]


def test_prepare_deep_analysis_context_always_routes_to_graph_table_content():
    assert agent.build_graph is not None
    assert agent.route_after_deep_analysis_draft({"status": "deep_analysis_draft_created"}) == "human_note_review"
