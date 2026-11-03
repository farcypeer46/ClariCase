"""Per-team TF-IDF retriever for few-shot example selection.

Builds one TF-IDF vectorizer and sparse matrix per team from the training
split, then returns the top-k most similar in-team examples for a query.
Index is cached to disk so predict runs don't rebuild it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.agentic_issue_classifier.config import (
    MAX_FEWSHOT_CHARS, RETRIEVER_CACHE, SEED,
)
from src.utils.data import load_split


@dataclass
class FewShotExample:
    text: str
    issue_id: str
    label: str


def _truncate(text: str, limit: int = MAX_FEWSHOT_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class PerTeamRetriever:
    def __init__(self, per_team: dict[str, dict]):
        self._per_team = per_team

    def topk(self, text: str, team_id: str, k: int) -> list[FewShotExample]:
        bucket = self._per_team.get(team_id)
        if not bucket or k <= 0:
            return []
        vec: TfidfVectorizer = bucket["vectorizer"]
        matrix: sparse.csr_matrix = bucket["matrix"]
        texts: list[str] = bucket["texts"]
        issue_ids: list[str] = bucket["issue_ids"]
        labels: list[str] = bucket["labels"]

        q = vec.transform([text])
        sims = (matrix @ q.T).toarray().ravel()
        k = min(k, len(texts))
        if k == 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [FewShotExample(text=_truncate(texts[i]),
                               issue_id=issue_ids[i],
                               label=labels[i]) for i in idx]

    @property
    def teams(self) -> list[str]:
        return sorted(self._per_team.keys())


def _issue_label_map() -> dict[str, str]:
    from src.agentic_issue_classifier.config import ISSUE_LABELS
    return ISSUE_LABELS


def build_retriever(per_team_cap: int = 2000,
                    cache_path: Path | None = RETRIEVER_CACHE,
                    force: bool = False) -> PerTeamRetriever:
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not force:
        return PerTeamRetriever(joblib.load(cache_path))

    df = load_split("train_full")[["complaint_text", "team_id", "issue_id"]]
    df = df.dropna(subset=["complaint_text", "team_id", "issue_id"])

    label_map = _issue_label_map()
    rng = np.random.default_rng(SEED)
    per_team: dict[str, dict] = {}
    for team_id, sub in df.groupby("team_id"):
        if len(sub) > per_team_cap:
            idx = rng.choice(len(sub), size=per_team_cap, replace=False)
            sub = sub.iloc[idx]
        texts = sub["complaint_text"].astype(str).tolist()
        issues = sub["issue_id"].astype(str).tolist()
        labels = [label_map.get(i, i) for i in issues]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                              sublinear_tf=True, lowercase=True)
        matrix = vec.fit_transform(texts)
        matrix = sparse.csr_matrix(matrix)
        # L2-normalize rows so (matrix @ q.T) is cosine similarity when q is
        # produced by the same vectorizer (TfidfVectorizer already L2-normalizes
        # by default; keep this check explicit).
        per_team[team_id] = {
            "vectorizer": vec,
            "matrix": matrix,
            "texts": texts,
            "issue_ids": issues,
            "labels": labels,
        }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(per_team, cache_path, compress=3)
    return PerTeamRetriever(per_team)


if __name__ == "__main__":
    r = build_retriever(force=True)
    print(f"[retriever] teams: {len(r.teams)}")
    for t in r.teams[:3]:
        n = len(r._per_team[t]["texts"])
        print(f"  {t}: {n} examples")
