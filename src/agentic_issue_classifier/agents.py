"""Classifier and verifier agents.

Each agent wraps a prompt builder and an LLM backend call. Agents return a
small result object the pipeline can log and reason over.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agentic_issue_classifier.config import TEAM_TO_ISSUES
from src.agentic_issue_classifier.llm import LLMBackend, LLMResult
from src.agentic_issue_classifier.prompts import (
    build_classifier_prompt, build_verifier_prompt,
)
from src.agentic_issue_classifier.retriever import FewShotExample


@dataclass
class AgentOutput:
    issue_id: str
    confidence: float
    raw: str
    prompt_chars: int


def _coerce_to_valid(text: str, valid: list[str]) -> str:
    stripped = (text or "").strip()
    if stripped in valid:
        return stripped
    for v in valid:
        if v in stripped:
            return v
    return "UNK"


class ClassifierAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def run(self, complaint: str, team_id: str,
            shots: list[FewShotExample]) -> AgentOutput:
        valid = TEAM_TO_ISSUES.get(team_id, [])
        prompt = build_classifier_prompt(complaint, team_id, shots)
        res: LLMResult = self.backend.generate(prompt, valid, max_new_tokens=12)
        issue_id = _coerce_to_valid(res.text, valid)
        return AgentOutput(issue_id=issue_id, confidence=res.confidence,
                           raw=res.raw, prompt_chars=len(prompt))


class VerifierAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def run(self, complaint: str, team_id: str,
            shots: list[FewShotExample], first_pass: str) -> AgentOutput:
        valid = TEAM_TO_ISSUES.get(team_id, [])
        prompt = build_verifier_prompt(complaint, team_id, shots, first_pass)
        res: LLMResult = self.backend.generate(prompt, valid, max_new_tokens=12)
        issue_id = _coerce_to_valid(res.text, valid)
        return AgentOutput(issue_id=issue_id, confidence=res.confidence,
                           raw=res.raw, prompt_chars=len(prompt))
