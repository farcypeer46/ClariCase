"""Prompt builders for the classifier and verifier agents."""
from __future__ import annotations

from src.agentic_issue_classifier.config import (
    ISSUE_LABELS, MAX_COMPLAINT_CHARS, TEAM_TO_ISSUES,
)
from src.agentic_issue_classifier.retriever import FewShotExample


def _truncate(text: str, limit: int = MAX_COMPLAINT_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _candidate_block(team_id: str) -> str:
    ids = TEAM_TO_ISSUES.get(team_id, [])
    lines = [f"- {iid}: {ISSUE_LABELS.get(iid, iid)}" for iid in ids]
    return "\n".join(lines)


def _shots_block(shots: list[FewShotExample]) -> str:
    if not shots:
        return "(no examples available)"
    parts = []
    for i, s in enumerate(shots, 1):
        parts.append(
            f"Example {i}\n"
            f"Complaint: {s.text}\n"
            f"Issue: {s.issue_id}  ({s.label})"
        )
    return "\n\n".join(parts)


CLASSIFIER_SYSTEM = (
    "You are an expert consumer-complaint analyst. For a given complaint "
    "handled by a known team, pick the single most appropriate issue ID "
    "from the team's issue list. Reply with the issue ID only."
)

VERIFIER_SYSTEM = (
    "You are a careful reviewer. A first-pass model proposed an issue ID for "
    "the complaint. Review the complaint, the candidate list, and additional "
    "examples. Produce the single best issue ID. Reply with the issue ID only."
)


def build_classifier_prompt(complaint: str, team_id: str,
                            shots: list[FewShotExample]) -> str:
    return (
        f"{CLASSIFIER_SYSTEM}\n\n"
        f"Team: {team_id}\n"
        f"Candidate issue IDs (choose exactly one):\n{_candidate_block(team_id)}\n\n"
        f"Reference examples from this team:\n{_shots_block(shots)}\n\n"
        f"New complaint:\n{_truncate(complaint)}\n\n"
        f"Answer with the issue ID only:"
    )


def build_verifier_prompt(complaint: str, team_id: str,
                          shots: list[FewShotExample],
                          first_pass: str) -> str:
    return (
        f"{VERIFIER_SYSTEM}\n\n"
        f"Team: {team_id}\n"
        f"First-pass proposal: {first_pass}\n"
        f"Candidate issue IDs (choose exactly one):\n{_candidate_block(team_id)}\n\n"
        f"Reference examples from this team:\n{_shots_block(shots)}\n\n"
        f"Complaint:\n{_truncate(complaint)}\n\n"
        f"Think briefly, then answer with the final issue ID only:"
    )
