"""Tests for src/jev_model. No network: HTTP is faked."""
import json

import numpy as np
import pandas as pd
import pytest

from src.jev_model import client as C
from src.jev_model import evaluate as E
from src.jev_model.config import TEAM_IDS, team_criteria


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses and records every request."""
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def post(self, url, json, timeout):
        self.calls.append({"url": url, "json": json})
        return self.responses.pop(0)


def _ok_body(top="T02"):
    probs = {t: 0.0 for t in TEAM_IDS}
    probs[top] = 0.9
    probs["T01"] = probs.get("T01", 0.0) + 0.1
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 1200},
            "answers": {"team": {"type": "choice", "choice": top,
                                 "confidence": 0.8, "probabilities": probs}}}


def test_criteria_describe_all_eleven_teams():
    crit = team_criteria()
    assert list(crit) == list(TEAM_IDS) and len(crit) == 11
    assert "Debt collection" in crit["T02"]
    assert all(desc.strip() for desc in crit.values())


def test_parse_reads_the_documented_choice_shape():
    ans = C.parse_team_answer(_ok_body("T05"))
    assert list(ans.probabilities) == list(TEAM_IDS)
    assert max(ans.probabilities, key=ans.probabilities.get) == "T05"
    assert ans.model == "jev-1.13.0" and ans.input_tokens == 1200


@pytest.mark.parametrize("body", [{}, {"answers": {}},
                                  {"answers": {"team": {"choice": "T01"}}}])
def test_parse_rejects_unexpected_shapes(body):
    with pytest.raises(C.JevError):
        C.parse_team_answer(body)


def test_parse_rejects_labels_outside_the_team_set():
    body = _ok_body()
    body["answers"]["team"]["probabilities"]["billing"] = 0.1
    with pytest.raises(C.JevError, match="outside the team set"):
        C.parse_team_answer(body)


def test_request_matches_the_api_reference():
    session = FakeSession([FakeResponse(200, _ok_body())])
    C.JevClient("k", model="jev-latest", session=session).classify_team("text")
    sent = session.calls[0]
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert session.headers["Authorization"] == "Bearer k"
    q = sent["json"]["questions"]["team"]
    assert sent["json"]["model"] == "jev-latest" and sent["json"]["state"] == "text"
    assert q["type"] == "choice" and set(q["criteria"]) == set(TEAM_IDS)


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(429, {}), FakeResponse(529, {}),
                           FakeResponse(200, _ok_body("T03"))])
    ans = C.JevClient("k", session=session).classify_team("text")
    assert len(session.calls) == 3
    assert max(ans.probabilities, key=ans.probabilities.get) == "T03"


def test_does_not_retry_a_rejected_request(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(422, {"detail": "questions missing"})])
    with pytest.raises(C.JevError, match="HTTP 422.*questions missing"):
        C.JevClient("k", session=session).classify_team("text")
    assert len(session.calls) == 1


def test_missing_key_is_a_clear_error():
    with pytest.raises(C.JevError, match="No API key"):
        C.JevClient("")


def test_main_without_key_exits_without_sending(monkeypatch, capsys):
    monkeypatch.setattr(E, "load_api_key", lambda: None)
    assert E.main([]) == 1
    assert "--mock" in capsys.readouterr().out


def test_sampled_text_is_pii_masked():
    df = pd.DataFrame({"complaint_id": ["1"], "team_id": ["T01"],
                       "complaint_text": ["Call me at 555-123-4567 please."]})
    out = E.sample_split(df, 1)
    assert "555-123-4567" not in out.loc[0, "model_text"]
    assert "XXXX" in out.loc[0, "model_text"]


class CountingMock(C.MockJevClient):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def classify_team(self, state):
        self.calls += 1
        return super().classify_team(state)


def test_cache_prevents_paying_twice(tmp_path):
    df = E.sample_split(pd.DataFrame({
        "complaint_id": [str(i) for i in range(20)], "team_id": ["T01"] * 20,
        "complaint_text": [f"My credit report has error number {i}." for i in range(20)]}), 20)
    client, cache = CountingMock(), tmp_path / "c.jsonl"
    first = E.classify_frame(client, df, cache)
    second = E.classify_frame(client, df, cache)
    assert client.calls == 20                       # second pass fully cached
    assert np.allclose(first, second) and first.shape == (20, 11)


def test_changed_text_is_requested_again(tmp_path):
    df = E.sample_split(pd.DataFrame({"complaint_id": ["1"], "team_id": ["T01"],
                                      "complaint_text": ["Old text here."]}), 1)
    client, cache = CountingMock(), tmp_path / "c.jsonl"
    E.classify_frame(client, df, cache)
    df.loc[0, ["model_text", "text_sha256"]] = ["New text here.", "different"]
    E.classify_frame(client, df, cache)
    assert client.calls == 2


def test_cost_estimate_counts_question_and_text():
    tokens, cost = E.estimate_cost(["x" * 400])
    assert tokens > 100 and cost > 0


# ---------------------------------------------------------------- variants
from src.jev_model.config import (STRUCTURED_TEAMS, cache_path, question_sha,
                                  team_question)


def test_structured_criteria_add_boundaries_and_examples_only_where_needed():
    crit = team_question("structured")["team"]["criteria"]
    for team in TEAM_IDS:
        if team in STRUCTURED_TEAMS:
            assert set(crit[team]) == {"description", "covers", "not_for", "examples"}
            assert 1 <= len(crit[team]["examples"]) <= 3
            assert all(e.strip() for e in crit[team]["examples"])
        else:
            assert isinstance(crit[team], str)
    # Instructions are identical, so only the descriptions differ.
    assert (team_question("structured")["team"]["instructions"]
            == team_question("zeroshot")["team"]["instructions"])


def test_examples_come_from_the_training_split():
    data = json.loads(open("src/jev_model/structured_examples.json").read())
    assert data["source_split"] == "train_cap_issue"
    assert set(data["examples"]) == set(STRUCTURED_TEAMS)


def test_each_variant_has_its_own_fingerprint_and_cache_file():
    assert question_sha("zeroshot") != question_sha("structured")
    assert cache_path("jev-1.14.0").name == "jev-1.14.0.jsonl"
    assert cache_path("jev-1.14.0", "structured").name == "jev-1.14.0__structured.jsonl"


def test_answers_are_not_reused_across_question_versions(tmp_path):
    df = E.sample_split(pd.DataFrame({"complaint_id": ["1"], "team_id": ["T01"],
                                      "complaint_text": ["Same text."]}), 1)
    client, cache = CountingMock(), tmp_path / "c.jsonl"
    E.classify_frame(client, df, cache, qsha="aaa", legacy_ok=False)
    E.classify_frame(client, df, cache, qsha="aaa", legacy_ok=False)
    assert client.calls == 1                       # same question: cached
    E.classify_frame(client, df, cache, qsha="bbb", legacy_ok=False)
    assert client.calls == 2                       # different question: asked


def test_answers_saved_before_fingerprints_count_only_for_zero_shot(tmp_path):
    df = E.sample_split(pd.DataFrame({"complaint_id": ["1"], "team_id": ["T01"],
                                      "complaint_text": ["Old run."]}), 1)
    cache = tmp_path / "c.jsonl"
    row = {"complaint_id": "1", "text_sha256": df.loc[0, "text_sha256"],
           "probabilities": {t: 1 / 11 for t in TEAM_IDS}, "model": "m",
           "input_tokens": 1}                      # no question_sha, like old rows
    cache.write_text(json.dumps(row) + "\n")
    loaded = E.load_cache(cache)
    assert E._pending(df, loaded, "zs", legacy_ok=True).empty
    assert len(E._pending(df, loaded, "st", legacy_ok=False)) == 1


# ---------------------------------------------------------------- rich variant
from src.jev_model.config import EXTRA_TEAMS, NOUL_QUESTIONS
from src.jev_model.stack import features


def test_parse_reads_yes_no_answers_alongside_the_team():
    body = _ok_body("T02")
    body["answers"]["collector"] = {"type": "noul", "noul": 0.93}
    body["answers"]["credit_report"] = {"type": "noul", "noul": 0.12}
    ans = C.parse_team_answer(body)
    assert ans.nouls == {"collector": 0.93, "credit_report": 0.12}
    assert max(ans.probabilities, key=ans.probabilities.get) == "T02"


def test_rich_question_adds_examples_only_for_weak_teams():
    from src.jev_model.config import WEAK_TEAMS
    q = team_question("rich")
    crit = q["team"]["criteria"]
    structured = team_question("structured")["team"]["criteria"]
    for t in TEAM_IDS:
        if t in WEAK_TEAMS:
            assert len(crit[t]["examples"]) == 5
        else:
            assert crit[t] == structured[t]   # unchanged from "structured"
    nouls = {k: v for k, v in q.items() if k != "team"}
    assert set(nouls) == set(NOUL_QUESTIONS)
    assert all(v["type"] == "noul" and v["instructions"] for v in nouls.values())


def test_structured_question_is_unchanged_so_its_cache_stays_valid():
    assert question_sha("structured") == "33b8258ba567"


def test_stack_features_flag_support_and_conflict():
    proba = np.zeros((2, 11)); proba[0, 0] = 1.0; proba[1, 1] = 1.0   # T01, T02
    nouls = [{"credit_report": 0.9, "collector": 0.8},   # T01 supported, but conflict
             {"credit_report": 0.1, "collector": 0.95}]  # T02 supported, no conflict
    X = features(proba, nouls)
    support, conflict = X[:, 2 + len(NOUL_QUESTIONS)], X[:, 3 + len(NOUL_QUESTIONS)]
    assert support.tolist() == [0.9, 0.95]
    assert conflict[0] == 0.8 and conflict[1] < 0.2
