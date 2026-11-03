"""End-to-end agentic pipeline: retrieve -> classify -> (verify).

Deterministic. A verifier pass is triggered when the classifier confidence
drops below `verify_threshold` OR the first-pass output is coerced to UNK.
The verifier is given a fresh, slightly larger set of few-shot examples so it
can disagree with the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from src.agentic_issue_classifier.agents import (
    AgentOutput, ClassifierAgent, VerifierAgent,
)
from src.agentic_issue_classifier.config import (
    NUM_FEW_SHOTS, VERIFIER_EXTRA_SHOTS, VERIFY_WHEN_UNCERTAIN,
)
from src.agentic_issue_classifier.llm import get_backend
from src.agentic_issue_classifier.retriever import PerTeamRetriever


@dataclass
class PipelineResult:
    issue_id: str
    confidence: float
    verified: bool
    classifier: dict
    verifier: dict | None

    def to_record(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "confidence": self.confidence,
            "verified": self.verified,
            "classifier": self.classifier,
            "verifier": self.verifier,
        }


class AgenticIssuePipeline:
    def __init__(self, retriever: PerTeamRetriever,
                 k_shots: int = NUM_FEW_SHOTS,
                 verify: bool = VERIFY_WHEN_UNCERTAIN,
                 verify_threshold: float = 0.6):
        backend = get_backend()
        self.retriever = retriever
        self.classifier = ClassifierAgent(backend)
        self.verifier = VerifierAgent(backend)
        self.k_shots = k_shots
        self.verify = verify
        self.verify_threshold = verify_threshold

    def predict(self, complaint: str, team_id: str) -> PipelineResult:
        shots = self.retriever.topk(complaint, team_id, self.k_shots)
        first: AgentOutput = self.classifier.run(complaint, team_id, shots)

        should_verify = self.verify and (
            first.confidence < self.verify_threshold or first.issue_id == "UNK"
        )
        if not should_verify:
            return PipelineResult(
                issue_id=first.issue_id, confidence=first.confidence,
                verified=False, classifier=asdict(first), verifier=None,
            )

        extra_k = self.k_shots + VERIFIER_EXTRA_SHOTS
        more_shots = self.retriever.topk(complaint, team_id, extra_k)
        second: AgentOutput = self.verifier.run(
            complaint, team_id, more_shots, first_pass=first.issue_id)
        final = second.issue_id if second.issue_id != "UNK" else first.issue_id
        final_conf = (second.confidence if second.issue_id != "UNK"
                      else first.confidence)
        return PipelineResult(
            issue_id=final, confidence=final_conf, verified=True,
            classifier=asdict(first), verifier=asdict(second),
        )
