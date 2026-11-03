"""LLM backend abstraction.

Three backends share a `generate(prompt, choices)` API where `choices` is the
list of valid issue IDs for the predicted team:

- groq: ChatGroq (`langchain-groq`), temperature=0. Default backend; uses
  `openai/gpt-oss-120b` via `GROQ_API_KEY`. Confidence proxy is shape-based
  (valid-choice hit → 1.0, else 0.5).
- openai: Chat Completions, temperature=0, with logprobs for a confidence
  proxy.
- hf_local: FLAN-T5 seq2seq with constrained decoding (prefix-allowed tokens).
  Deterministic greedy; good offline fallback.

Each `generate` returns (text, confidence, raw) so the pipeline can reflect on
low-confidence outputs without reaching into backend internals.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from src.agentic_issue_classifier.config import (
    GROQ_BASE_URL, GROQ_MODEL_NAME, HF_MODEL_NAME, LLM_BACKEND, OPENAI_MODEL_NAME,
)


@dataclass
class LLMResult:
    text: str
    confidence: float
    raw: str


class LLMBackend(Protocol):
    def generate(self, prompt: str, choices: list[str],
                 max_new_tokens: int = 16) -> LLMResult: ...


class HFLocalBackend:
    def __init__(self, model_name: str = HF_MODEL_NAME):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float32)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self._device = next(self.model.parameters()).device

    def _prefix_fn(self, choices: list[str]):
        # Build a token-level trie over `choices` so the decoder can only emit
        # tokens that continue to a valid issue ID. EOS is permitted once a
        # choice is fully emitted.
        eos = self.tok.eos_token_id
        encoded = [self.tok(c, add_special_tokens=False).input_ids
                   for c in choices]

        def allowed(_batch_id: int, input_ids):
            # Decoder input starts with pad (decoder_start_token_id for T5).
            # input_ids[0] is that pad; actual generated tokens follow.
            emitted = input_ids.tolist()[1:]
            next_ok: set[int] = set()
            for ids in encoded:
                if len(emitted) < len(ids) and ids[:len(emitted)] == emitted:
                    next_ok.add(ids[len(emitted)])
                elif ids == emitted:
                    next_ok.add(eos)
            return list(next_ok) if next_ok else [eos]

        return allowed

    def generate(self, prompt: str, choices: list[str],
                 max_new_tokens: int = 16) -> LLMResult:
        torch = self._torch
        inputs = self.tok(prompt, return_tensors="pt", truncation=True,
                          max_length=2048).to(self._device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                prefix_allowed_tokens_fn=self._prefix_fn(choices),
                return_dict_in_generate=True,
                output_scores=True,
            )
        seq = out.sequences[0]
        text = self.tok.decode(seq, skip_special_tokens=True).strip()
        # Confidence proxy: mean of top-token log-softmax across generated steps.
        logps = []
        for step, scores in enumerate(out.scores):
            tok_id = seq[step + 1].item()  # +1 skips decoder_start pad
            step_logp = torch.log_softmax(scores[0].float(), dim=-1)[tok_id]
            logps.append(step_logp.item())
        conf = math.exp(sum(logps) / len(logps)) if logps else 0.0
        return LLMResult(text=text, confidence=conf, raw=text)


class OpenAIBackend:
    def __init__(self, model_name: str = OPENAI_MODEL_NAME):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set; export it or pick AIC_LLM_BACKEND=hf_local")
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name

    @staticmethod
    def _extract_choice(text: str, choices: list[str]) -> str:
        for c in choices:
            if re.search(rf"\b{re.escape(c)}\b", text):
                return c
        return text.strip().splitlines()[0].strip()

    def generate(self, prompt: str, choices: list[str],
                 max_new_tokens: int = 16) -> LLMResult:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_new_tokens,
            logprobs=True,
            top_logprobs=1,
        )
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        picked = self._extract_choice(raw, choices)
        conf = 0.0
        if choice.logprobs and choice.logprobs.content:
            lps = [t.logprob for t in choice.logprobs.content]
            if lps:
                conf = math.exp(sum(lps) / len(lps))
        return LLMResult(text=picked, confidence=conf, raw=raw)


class GroqBackend:
    """Groq via `langchain-groq`'s ChatGroq. Install with `pip install langchain-groq`.

    Confidence proxy is shape-based (1.0 when the answer parses to a valid
    in-team issue ID, else 0.5) because ChatGroq doesn't surface per-token
    logprobs uniformly across models — good enough to gate the verifier.
    """

    def __init__(self, model_name: str = GROQ_MODEL_NAME,
                 base_url: str | None = GROQ_BASE_URL):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set; export it or pick a different "
                "AIC_LLM_BACKEND (openai | hf_local)")
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise RuntimeError(
                "langchain-groq not installed; run `pip install langchain-groq`"
            ) from e
        self.model_name = model_name
        self._ChatGroq = ChatGroq
        self._api_key = key
        self._base_url = base_url
        # Rebuild per call so max_tokens can vary; construction is cheap.

    @staticmethod
    def _extract_choice(text: str, choices: list[str]) -> str:
        if not text:
            return ""
        for c in choices:
            if re.search(rf"\b{re.escape(c)}\b", text):
                return c
        return text.strip().splitlines()[-1].strip()

    def generate(self, prompt: str, choices: list[str],
                 max_new_tokens: int = 16) -> LLMResult:
        from langchain_core.messages import HumanMessage
        # Reasoning models (gpt-oss, etc.) consume the max_tokens budget on
        # hidden reasoning before emitting content. Floor at 1024 so a short
        # label (~5 tokens) survives after reasoning.
        token_budget = max(max_new_tokens, 1024)
        kwargs = dict(model=self.model_name, temperature=0,
                      max_tokens=token_budget, api_key=self._api_key,
                      reasoning_effort="low")
        if self._base_url:
            kwargs["base_url"] = self._base_url
        llm = self._ChatGroq(**kwargs)
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = (getattr(resp, "content", "") or "").strip()
        picked = self._extract_choice(raw, choices)
        conf = 1.0 if picked in choices else 0.5
        return LLMResult(text=picked, confidence=conf, raw=raw)


@lru_cache(maxsize=1)
def get_backend() -> LLMBackend:
    name = LLM_BACKEND.lower()
    if name == "groq":
        return GroqBackend()
    if name == "openai":
        return OpenAIBackend()
    if name == "hf_local":
        return HFLocalBackend()
    raise ValueError(f"Unknown AIC_LLM_BACKEND={LLM_BACKEND!r}")
