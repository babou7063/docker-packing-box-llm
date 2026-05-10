# -*- coding: UTF-8 -*-
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

from pboxllm import LLMBackend, FeatureFormatter, PromptStrategy, proba_from_top_logprobs


__all__ = ["LLMClassifier"]


class LLMClassifier(BaseEstimator):
    """LLM-based packing classifier using llama-cpp-python (zero-shot).

    This classifier mimics the sklearn fit/predict interface expected by pbox.
    It uses a local large language model (GGUF format, via llama-cpp-python)
    to determine whether a PE executable is packed, based on a human-readable
    text representation of selected binary features.

    No gradient-based training occurs. ``fit`` loads the model into memory;
    ``predict`` runs zero-shot inference for each sample.
    """

    classes_ = np.array([0, 1])
    _few_shot_seed = 42
    _required_params = (
        "model_file",
        "model_repo",
        "prompt_file",
        "feature_names",
        "n_ctx",
        "n_threads",
        "max_tokens",
    )

    def __init__(
        self,
        model_file=None,
        model_repo=None,
        prompt_file=None,
        feature_names=None,
        n_ctx=None,
        n_threads=None,
        max_tokens=None,
        temperature=0.0,
        top_p=1.0,
        packed_label="packed",
        not_packed_label="not-packed",
        representation_style="flat",
        few_shot_mode=False,
        few_shot_count=4,
        trace_outputs=False,
        trace_dir=None,
        parse_policy="p_packed_fallback",
        on_backend_error="unknown",
    ):
        self.model_file = model_file
        self.model_repo = model_repo
        self.prompt_file = prompt_file
        self.feature_names = feature_names
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.packed_label = packed_label
        self.not_packed_label = not_packed_label
        self.representation_style = representation_style
        self.few_shot_mode = few_shot_mode
        self.few_shot_count = few_shot_count
        self.trace_outputs = trace_outputs
        self.trace_dir = trace_dir
        self.parse_policy = parse_policy
        self.on_backend_error = on_backend_error

    def __sklearn_is_fitted__(self):
        return hasattr(self, "backend_")

    def fit(self, X, y=None):
        missing = [p for p in self._required_params if getattr(self, p) in [None, ""]]
        if missing:
            raise ValueError(
                "[LLMClassifier] Missing required parameter(s): "
                + ", ".join(missing)
                + ". Configure them in algorithms.yml (LLM category)."
            )
        self.backend_ = LLMBackend(self.model_file, self.model_repo, self.n_ctx, self.n_threads)
        self.formatter_ = FeatureFormatter(self.feature_names, representation_style=self.representation_style)
        self.strategy_ = PromptStrategy(
            self.prompt_file,
            packed_label=self.packed_label,
            not_packed_label=self.not_packed_label,
            parse_policy=self.parse_policy,
        )
        self.few_shot_examples_ = self._build_few_shot_examples(X, y)
        self._trace_run_id = None
        self._trace_path = None
        self.backend_.load()
        return self

    def predict(self, X):
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict.")
        results = []
        for i in range(len(X)):
            row = X.iloc[i] if hasattr(X, "iloc") else X[i]
            text = self.formatter_.format(row)
            prompt = self.strategy_.build_prompt(text, few_shot_examples=self.few_shot_examples_)
            if i == 0:
                print("\n===== DEBUG PROMPT (first sample) =====\n")
                print(prompt)
                print("\n===== END DEBUG PROMPT =====\n")
            backend_error = None
            try:
                raw = self.backend_.generate(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            except Exception as exc:
                raw = ""
                backend_error = {"type": type(exc).__name__, "message": str(exc)}
            if i == 0:
                print("\n===== DEBUG RAW LLM RESPONSE (first sample) =====\n")
                print(raw)
                print("\n===== DEBUG RAW LLM RESPONSE REPR (first sample) =====\n")
                print(repr(raw))
                print("\n===== END DEBUG RAW LLM RESPONSE =====\n")
            parsed_meta = self.strategy_.parse_with_meta(raw)
            parsed = int(parsed_meta["label"])
            if backend_error is not None and str(self.on_backend_error).lower() == "unknown":
                parsed = -1
                parsed_meta["reason"] = "backend_error"
            self._trace_sample(
                method="predict",
                sample_index=i,
                prompt=prompt,
                raw_response=raw,
                parsed_label=int(parsed),
                parse_meta=parsed_meta,
                error=backend_error,
            )
            results.append(parsed)
        return np.array(results, dtype=int)

    def predict_proba(self, X):
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict_proba.")
        probas = []
        for i in range(len(X)):
            out = None
            raw = ""
            backend_error = None
            row = X.iloc[i] if hasattr(X, "iloc") else X[i]
            text = self.formatter_.format(row)
            prompt = self.strategy_.build_prompt(text, few_shot_examples=self.few_shot_examples_)
            try:
                out = self.backend_.generate_with_logprobs(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                if i == 0:
                    print("\n===== DEBUG RAW LLM RESPONSE PROBA PATH (first sample) =====\n")
                    print(out.get("text", ""))
                    print("\n===== DEBUG RAW LLM RESPONSE PROBA REPR (first sample) =====\n")
                    print(repr(out.get("text", "")))
                    print("\n===== END DEBUG RAW LLM RESPONSE PROBA PATH =====\n")
                proba = proba_from_top_logprobs(out.get("top_logprobs"))
                if proba is None:
                    pred = self.strategy_.parse(out.get("text", ""))
                    if pred == 1:
                        proba = np.array([0.0, 1.0], dtype=float)
                    elif pred == 0:
                        proba = np.array([1.0, 0.0], dtype=float)
                    else:
                        proba = np.array([0.5, 0.5], dtype=float)
            except ValueError:
                # Some GGUF builds don't expose logprobs (logits_all=False).
                # Fallback: generate plain text and map it to packedness.
                try:
                    raw = self.backend_.generate(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                except Exception as exc:
                    raw = ""
                    backend_error = {"type": type(exc).__name__, "message": str(exc)}
                parsed_meta = self.strategy_.parse_with_meta(raw)
                pred = parsed_meta["label"]
                if backend_error is not None and str(self.on_backend_error).lower() == "unknown":
                    pred = -1
                if pred == 1:
                    proba = np.array([0.0, 1.0], dtype=float)
                elif pred == 0:
                    proba = np.array([1.0, 0.0], dtype=float)
                else:
                    proba = np.array([0.5, 0.5], dtype=float)
            except Exception as exc:
                backend_error = {"type": type(exc).__name__, "message": str(exc)}
                proba = np.array([0.5, 0.5], dtype=float)
            probas.append(proba)
        return np.vstack(probas)

    def _build_few_shot_examples(self, X, y):
        if not self.few_shot_mode:
            return []
        if y is None:
            return []
        labels = np.asarray(y)
        if labels.size == 0:
            return []
        total = int(self.few_shot_count or 0)
        if total <= 0:
            return []
        rng = np.random.RandomState(self._few_shot_seed)
        packed_idx = np.where(labels == 1)[0]
        not_packed_idx = np.where(labels == 0)[0]
        half = max(1, total // 2)
        selected = []
        if packed_idx.size:
            selected.extend(rng.choice(packed_idx, size=min(half, packed_idx.size), replace=False).tolist())
        if not_packed_idx.size:
            selected.extend(rng.choice(not_packed_idx, size=min(total - len(selected), not_packed_idx.size), replace=False).tolist())
        remaining = total - len(selected)
        if remaining > 0:
            all_idx = np.arange(labels.size)
            left = np.setdiff1d(all_idx, np.array(selected, dtype=int), assume_unique=False)
            if left.size:
                selected.extend(rng.choice(left, size=min(remaining, left.size), replace=False).tolist())
        examples = []
        for idx in selected:
            row = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            examples.append({
                "features_text": self.formatter_.format(row),
                "label": int(labels[idx]),
            })
        return examples

    def _resolve_trace_path(self):
        if self._trace_path is not None:
            return self._trace_path
        base_dir = self.trace_dir or os.path.expanduser("~/.packing-box/cache/llm-traces")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        if self._trace_run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._trace_run_id = f"{ts}-pid{os.getpid()}"
        model_stub = str(self.model_file or "model").replace("/", "_")
        algo_stub = str(self.representation_style or "flat").replace("/", "_")
        prompt_stub = str(self.prompt_file or "prompt").replace("/", "_")
        self._trace_path = str(Path(base_dir) / f"{self._trace_run_id}-{algo_stub}-{prompt_stub}-{model_stub}.jsonl")
        return self._trace_path

    def _trace_sample(
        self,
        method,
        sample_index,
        prompt,
        raw_response,
        parsed_label,
        proba=None,
        parse_meta=None,
        error=None,
    ):
        if not self.trace_outputs:
            return
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": self._trace_run_id or "pending",
            "method": method,
            "sample_index": int(sample_index),
            "model_file": self.model_file,
            "model_repo": self.model_repo,
            "prompt_file": self.prompt_file,
            "representation_style": self.representation_style,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "max_tokens": self.max_tokens,
            "temperature": getattr(self, "temperature", None),
            "top_p": getattr(self, "top_p", None),
            "parsed_label": int(parsed_label),
            "parse_meta": parse_meta or {},
            "raw_response": str(raw_response),
            "proba": proba,
            "error": error,
            "prompt": prompt,
        }
        path = self._resolve_trace_path()
        event["trace_path"] = path
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
