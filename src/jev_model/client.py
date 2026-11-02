"""Minimal client for Jev's System One endpoint, plus an offline mock.

The key is read from TYPESAFE_API_KEY or `[typesafe] key` in
.streamlit/secrets.toml (gitignored). It is never printed or logged.
"""
from __future__ import annotations

import hashlib
import os
import random
import time
import tomllib
from dataclasses import dataclass

import numpy as np
import requests

from src.jev_model.config import (ENDPOINT, MAX_RETRIES, MODEL, QUESTION_NAME,
                                  TEAM_IDS, TIMEOUT_SEC, team_question)
from src.utils.config import ROOT

RETRY_STATUSES = {429, 500, 502, 503, 529}


class JevError(RuntimeError):
    """A request Jev rejected or that kept failing after retries."""


@dataclass
class TeamAnswer:
    probabilities: dict[str, float]   # team ID -> probability
    model: str                        # version that actually answered
    input_tokens: int


def load_api_key() -> str | None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        with open(secrets, "rb") as f:
            return tomllib.load(f).get("typesafe", {}).get("key")
    return None


def parse_team_answer(body: dict) -> TeamAnswer:
    """Read the Choice answer for the team question from a response body."""
    try:
        answer = body["answers"][QUESTION_NAME]
        probs = answer["probabilities"]
    except (KeyError, TypeError) as exc:
        raise JevError(f"Unexpected response shape: {sorted(body)}") from exc
    unknown = set(probs) - set(TEAM_IDS)
    if unknown:
        raise JevError(f"Response has labels outside the team set: {unknown}")
    return TeamAnswer(
        probabilities={t: float(probs.get(t, 0.0)) for t in TEAM_IDS},
        model=str(body.get("model", "")),
        input_tokens=int(body.get("usage", {}).get("input_tokens", 0)),
    )


class JevClient:
    def __init__(self, api_key: str, model: str = MODEL,
                 endpoint: str = ENDPOINT, session: requests.Session | None = None):
        if not api_key:
            raise JevError("No API key: set TYPESAFE_API_KEY or add "
                           "[typesafe] key to .streamlit/secrets.toml.")
        self.model = model
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}",
                                     "Content-Type": "application/json"})
        self._question = team_question()

    def raw(self, state: str) -> dict:
        """One request; returns the parsed JSON body. Retries 429/529/5xx."""
        payload = {"model": self.model, "state": state,
                   "questions": self._question}
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.session.post(self.endpoint, json=payload,
                                         timeout=TIMEOUT_SEC)
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    raise JevError(f"Network error: {exc}") from exc
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    # The body names the problem (e.g. which field failed).
                    raise JevError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            # Exponential backoff with jitter, as the docs advise for 429/529.
            time.sleep(min(30.0, 2 ** attempt) + random.random())
        raise JevError("unreachable")

    def classify_team(self, state: str) -> TeamAnswer:
        return parse_team_answer(self.raw(state))


class MockJevClient:
    """Offline stand-in with the same interface, for testing the pipeline.

    Returns deterministic pseudo-probabilities derived from the text. Its
    scores mean nothing; every artifact it produces is labelled as mock.
    """
    model = "mock"

    def classify_team(self, state: str) -> TeamAnswer:
        seed = int.from_bytes(hashlib.sha256(state.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.full(len(TEAM_IDS), 0.3))
        return TeamAnswer(dict(zip(TEAM_IDS, p.tolist())), self.model,
                          len(state) // 4)
