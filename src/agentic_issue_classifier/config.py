"""Static configuration for the agentic issue classifier.

Env vars:
    AIC_LLM_BACKEND   groq | openai | hf_local    (default groq)
    AIC_GROQ_MODEL    Groq model id               (default openai/gpt-oss-120b)
    AIC_OPENAI_MODEL  OpenAI model name           (default gpt-4o-mini)
    AIC_HF_MODEL      HF seq2seq model name       (default google/flan-t5-large)
    GROQ_API_KEY      required when backend=groq
    OPENAI_API_KEY    required when backend=openai
"""
from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import CACHE, ROOT, RANDOM_SEED
from src.utils.labels import issue_labels, team_to_issues

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

LLM_BACKEND = os.environ.get("AIC_LLM_BACKEND", "groq")
GROQ_MODEL_NAME = os.environ.get("AIC_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.environ.get("AIC_GROQ_BASE_URL") or None
OPENAI_MODEL_NAME = os.environ.get("AIC_OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL_NAME = os.environ.get("AIC_HF_MODEL", "google/flan-t5-large")

NUM_FEW_SHOTS = 8
MAX_COMPLAINT_CHARS = 1200
MAX_FEWSHOT_CHARS = 500

VERIFY_WHEN_UNCERTAIN = True
VERIFIER_EXTRA_SHOTS = 4

LOG_DIR = ROOT / "logs" / "agentic_issue_classifier"
CACHE_DIR = CACHE / "agentic_issue_classifier"
RETRIEVER_CACHE = CACHE_DIR / "retriever_index.joblib"

SEED = RANDOM_SEED

ISSUE_LABELS: dict[str, str] = issue_labels
TEAM_TO_ISSUES: dict[str, list[str]] = team_to_issues

for _p in (LOG_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)
