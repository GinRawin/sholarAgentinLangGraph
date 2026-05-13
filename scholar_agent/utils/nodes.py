from __future__ import annotations

from pathlib import Path

from langgraph.types import interrupt

from scholar_agent.config import DEEP_ANALYSIS_TEMPLATE, SUMMARY_TEMPLATE
from scholar_agent.utils.state import ResearchAgentState, SummaryResult
from scholar_agent.utils.tools import build_pdf_attachment, collect_known_terms, default_pdf_root, extract_pdf_title, get_llm, get_repository, iter_pdf_paths, read_note, read_related_notes, serialize_paper, write_draft_note, write_final_note


def initialize_memory_node(state: ResearchAgentState) -> ResearchAgentState:
    del state
    repository = get_repository()
    repository.init_db()
    return {"status": "memory_ready"}


def scan_library_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    root = Path(state.get("pdf_root") or default_pdf_root())
    scanned_count = 0

    for pdf_path in iter_pdf_paths(root):
        title = extract_pdf_title(pdf_path)
        repository.upsert_scanned_paper(title=title, pdf_path=pdf_path)
        scanned_count += 1

    return {
        "status": "library_scanned",
        "pdf_root": str(root),
        "scanned_count": scanned_count,
    }


def select_unsummarized_paper_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    pending = repository.list_pending_summary(limit=1)
    paper = pending[0] if pending else None

    if paper is None:
        return {
            "status": "no_unsummarized_paper",
            "title": "",
            "selected_paper": {},
        }

    return {
        "status": "unsummarized_paper_selected",
        "title": paper.title,
        "selected_paper": serialize_paper(paper),
    }


def generate_summary_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    llm = get_llm()
    title = state.get("title")
    if not title:
        return {"status": "error", "error": "title is required for summary generation"}

    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    known_keywords, known_categories = collect_known_terms(repository)
    pdf_attachment = build_pdf_attachment(paper.pdf_path)
    prompt = SUMMARY_TEMPLATE.format(
        known_keywords=", ".join(known_keywords) or "无",
        known_categories=", ".join(known_categories) or "无",
    )
    result = llm.summarize_paper(
        title=paper.title,
        prompt=prompt,
        known_keywords=known_keywords,
        known_categories=known_categories,
        pdf_attachment=pdf_attachment,
    )
    repository.save_summary(result)

    return _summary_result_to_state(result) | {
        "status": "summary_generated",
        "known_keywords": known_keywords,
        "known_categories": known_categories,
    }


def load_review_queue_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    limit = int(state.get("limit") or 5)
    papers = repository.list_unread_summaries(limit=limit)
    return {
        "status": "review_queue_loaded",
        "review_queue": [serialize_paper(paper) for paper in papers],
    }


def human_summary_review_node(state: ResearchAgentState) -> ResearchAgentState:
    queue = state.get("review_queue", [])
    if not queue:
        return {"status": "no_summary_waiting_for_user"}

    paper = queue[0]
    payload = interrupt(
        {
            "kind": "summary_decision",
            "paper": paper,
            "prompt": "阅读摘要后请选择 deep_analysis 或 skip。",
            "allowed_decisions": ["deep_analysis", "skip"],
        }
    )
    user_decision = _normalize_summary_decision(payload)

    return {
        "status": "summary_reviewed_by_user",
        "title": paper["title"],
        "selected_paper": paper,
        "user_decision": user_decision,
    }


def record_summary_decision_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    title = state.get("title")
    if not title:
        return {
            "status": "error",
            "error": "title is required for user decision",
        }

    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    repository.mark_user_read(title)
    decision = state.get("user_decision", "skip")
    return {
        "status": "summary_marked_read",
        "title": title,
        "user_decision": decision,
        "selected_paper": serialize_paper(paper),
    }


def prepare_deep_analysis_context_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    title = state.get("title")
    if not title:
        return {"status": "error", "error": "title is required for deep analysis"}

    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    related_papers = repository.list_by_categories(paper.categories, exclude_title=paper.title)
    ranked_related_papers = sorted(
        related_papers,
        key=lambda related: (
            _keyword_overlap_count(paper.keywords, related.keywords),
            related.updated_at.timestamp() if related.updated_at else 0.0,
            related.title,
        ),
        reverse=True,
    )
    related_note_paths = [paper.note_path for paper in ranked_related_papers if paper.note_path][:5]
    related_notes = read_related_notes(related_note_paths)
    prompt = DEEP_ANALYSIS_TEMPLATE.format(
        related_notes=related_notes or "无",
    )
    repository.mark_user_read(paper.title)

    return {
        "status": "deep_analysis_context_ready",
        "title": paper.title,
        "selected_paper": serialize_paper(paper),
        "related_note_paths": related_note_paths,
        "related_notes": related_notes,
        "deep_analysis_prompt": prompt,
    }


def _keyword_overlap_count(source_keywords: list[str], candidate_keywords: list[str]) -> int:
    source = {
        str(keyword).strip().casefold()
        for keyword in source_keywords
        if str(keyword).strip()
    }
    if not source:
        return 0

    candidate = {
        str(keyword).strip().casefold()
        for keyword in candidate_keywords
        if str(keyword).strip()
    }
    return len(source.intersection(candidate))


def draft_deep_analysis_note_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    llm = get_llm()
    title = state.get("title")
    prompt = state.get("deep_analysis_prompt")
    if not title or not prompt:
        return {"status": "error", "error": "title and deep_analysis_prompt are required"}

    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    draft = llm.draft_deep_note(
        title=title,
        prompt=prompt,
        related_note_paths=state.get("related_note_paths", []),
        pdf_attachment=build_pdf_attachment(paper.pdf_path),
    )
    session_id = repository.create_deep_analysis_session(paper_title=title)
    note_path = write_draft_note(paper=paper, draft_note=draft)
    return {
        "status": "deep_analysis_draft_created",
        "draft_note": draft,
        "note_path": str(note_path),
        "deep_analysis_session_id": session_id,
    }


def human_note_review_node(state: ResearchAgentState) -> ResearchAgentState:
    note_path = state.get("note_path")
    draft_note = ""
    if note_path:
        try:
            draft_note = read_note(note_path)
        except FileNotFoundError:
            draft_note = state.get("draft_note", "")
    else:
        draft_note = state.get("draft_note", "")
    payload = interrupt(
        {
            "kind": "note_review",
            "title": state.get("title", ""),
            "draft_note": draft_note,
            "latest_answer": state.get("latest_answer", ""),
            "qa_history": state.get("qa_history", []),
            "prompt": (
                "如需继续追问，请直接输入问题；输入 done 表示问答结束并据此修订笔记；"
                "输入 confirm 表示直接确认终稿。"
            ),
            "allowed_actions": ["ask_question", "done", "confirm"],
        }
    )
    message = str(payload).strip()
    first_word = message.split(maxsplit=1)[0].lower() if message else ""
    if first_word == "confirm":
        final_note = message[len("confirm") :].strip() or draft_note
        return {
            "status": "note_confirmed_by_user",
            "note_review_action": "confirm",
            "final_note": final_note,
        }
    if first_word == "done":
        return {
            "status": "note_questions_completed_by_user",
            "note_review_action": "finish_qa",
        }

    return {
        "status": "note_question_requested_by_user",
        "note_review_action": "ask_question",
        "latest_question": message,
        "user_message": message,
    }


def question_and_answer_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    llm = get_llm()
    title = state.get("title")
    note_path = state.get("note_path")
    latest_question = state.get("latest_question") or state.get("user_message")
    if not title or not latest_question:
        return {
            "status": "error",
            "error": "title and latest_question are required",
        }

    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    current_note = state.get("draft_note", "")
    if note_path:
        try:
            current_note = read_note(note_path)
        except FileNotFoundError:
            current_note = state.get("draft_note", "")

    answer = llm.answer_note_question(
        title=title,
        current_note=current_note,
        user_question=latest_question,
        pdf_attachment=build_pdf_attachment(paper.pdf_path),
    )
    qa_history = list(state.get("qa_history", []))
    qa_history.append(
        {
            "question": latest_question,
            "answer": answer,
        }
    )
    return {
        "status": "question_answered",
        "latest_question": latest_question,
        "latest_answer": answer,
        "question_answer": answer,
        "qa_history": qa_history,
    }


def revise_deep_analysis_note_node(state: ResearchAgentState) -> ResearchAgentState:
    llm = get_llm()
    title = state.get("title")
    note_path = state.get("note_path")
    if not title or not note_path:
        return {
            "status": "error",
            "error": "title and note_path are required",
        }

    try:
        current_note = read_note(note_path)
    except FileNotFoundError:
        return {
            "status": "error",
            "error": f"Note file not found: {note_path}",
        }

    qa_history = state.get("qa_history", [])
    if not qa_history:
        return {
            "status": "deep_analysis_draft_revised",
            "draft_note": current_note,
            "note_path": str(note_path),
        }

    conversation = "\n\n".join(
        (
            f"第 {index} 轮问答\n"
            f"问题：{turn['question'].strip()}\n"
            f"回答：{turn['answer'].strip()}"
        )
        for index, turn in enumerate(qa_history, start=1)
    )
    revision_prompt = (
        "下面是用户围绕论文细节的问答记录。"
        "请基于这些已经澄清的信息更新整篇笔记，返回完整修订后的笔记。\n\n"
        f"{conversation}"
    )
    revised = llm.revise_note(
        title=title,
        current_note=current_note,
        user_message=revision_prompt,
    )
    repository = get_repository()
    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    write_draft_note(paper=paper, draft_note=revised)
    return {
        "status": "deep_analysis_draft_revised",
        "draft_note": revised,
        "note_path": str(note_path),
    }


def save_final_note_node(state: ResearchAgentState) -> ResearchAgentState:
    repository = get_repository()
    title = state.get("title")
    note_path = state.get("note_path")
    final_note = state.get("final_note") or state.get("draft_note")
    if not final_note and note_path:
        try:
            final_note = read_note(note_path)
        except FileNotFoundError:
            final_note = None
    if not title or not final_note:
        return {"status": "error", "error": "title and final_note are required"}

    paper = repository.get_paper(title)
    if paper is None:
        return {"status": "error", "error": f"Paper not found: {title}"}

    note_path = write_final_note(paper=paper, final_note=final_note)
    repository.mark_deep_analyzed(title=paper.title, note_path=note_path)
    session_id = state.get("deep_analysis_session_id")
    if session_id:
        repository.confirm_deep_analysis_session(session_id=session_id, note_path=note_path)
    return {
        "status": "final_note_saved",
        "note_path": str(note_path),
    }


def _summary_result_to_state(result: SummaryResult) -> ResearchAgentState:
    return {
        "title": result.title,
        "summary": result.summary,
        "keywords": result.keywords,
        "categories": result.categories,
    }


def _normalize_summary_decision(payload: object) -> str:
    if isinstance(payload, int):
        return {0: "deep_analysis", 1: "skip"}.get(payload, "skip")

    text = str(payload).strip().lower()
    if text == "0":
        return "deep_analysis"
    if text == "1":
        return "skip"
    if text in {"deep_analysis", "skip"}:
        return text
    return "skip"
