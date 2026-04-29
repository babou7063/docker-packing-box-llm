# -*- coding: UTF-8 -*-
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
        )
        self.few_shot_examples_ = self._build_few_shot_examples(X, y)
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
            raw = self.backend_.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if i == 0:
                print("\n===== DEBUG RAW LLM RESPONSE (first sample) =====\n")
                print(raw)
                print("\n===== DEBUG RAW LLM RESPONSE REPR (first sample) =====\n")
                print(repr(raw))
                print("\n===== END DEBUG RAW LLM RESPONSE =====\n")
            results.append(self.strategy_.parse(raw))
        return np.array(results, dtype=int)

    def predict_proba(self, X):
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict_proba.")
        probas = []
        for i in range(len(X)):
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
                raw = self.backend_.generate(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                pred = self.strategy_.parse(raw)
                if pred == 1:
                    proba = np.array([0.0, 1.0], dtype=float)
                elif pred == 0:
                    proba = np.array([1.0, 0.0], dtype=float)
                else:
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
