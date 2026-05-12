from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypedDict


UserDecision = Literal["deep_analysis", "skip"]
NoteReviewAction = Literal["revise", "confirm"]


@dataclass(slots=True)
class Paper:
    title: str
    pdf_path: str
    conference: str | None = None
    publication_time: str | None = None
    summary_generated: bool = False
    user_read: bool = False
    deep_analyzed: bool = False
    note_path: str | None = None
    summary: str | None = None
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class SummaryResult:
    title: str
    summary: str
    keywords: list[str]
    categories: list[str]
    conference: str | None = None
    publication_time: str | None = None


class PaperRecord(TypedDict, total=False):
    title: str
    conference: str | None
    publication_time: str | None
    summary_generated: bool
    user_read: bool
    deep_analyzed: bool
    note_path: str | None
    pdf_path: str
    summary: str | None
    keywords: list[str]
    categories: list[str]


class ResearchAgentState(TypedDict, total=False):
    status: str
    error: str

    pdf_root: str
    scanned_count: int

    title: str
    selected_paper: PaperRecord
    review_queue: list[PaperRecord]

    known_keywords: list[str]
    known_categories: list[str]
    summary: str
    keywords: list[str]
    categories: list[str]

    user_decision: UserDecision
    note_review_action: NoteReviewAction
    user_message: str
    question_answer: str
    final_note: str

    related_note_paths: list[str]
    related_notes: str
    deep_analysis_prompt: str
    draft_note: str
    note_path: str
    deep_analysis_session_id: int
