# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path


_LLM_PATH = Path(__file__).resolve().parents[2] / "pbox" / "core" / "model" / "algorithm" / "llm.py"
_SPEC = importlib.util.spec_from_file_location("llm_module_under_test", _LLM_PATH)
llm_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(llm_module)


class Backend:
    def __init__(self, *args, **kwargs):
        self.generated_prompts = []
        self._llm = None

    def load(self):
        return None

    def generate(self, prompt, max_tokens=64, temperature=0.0, top_p=1.0):
        self.generated_prompts.append(prompt)
        return "packed"

    def generate_with_logprobs(self, prompt, max_tokens=64, temperature=0.0, top_p=1.0, logprobs=10):
        self.generated_prompts.append(prompt)
        return {
            "text": "packed",
            "top_logprobs": {
                " packed": -0.1,
                " not": -2.2,
            },
        }


class Formatter:
    def __init__(self, feature_names, representation_style="flat"):
        self.feature_names = feature_names
        self.representation_style = representation_style

    def format(self, row):
        return " | ".join(f"{k}={row[k]}" for k in self.feature_names)


class Strategy:
    def __init__(self, prompt_file, **kwargs):
        self.prompt_file = prompt_file
        self.received_few_shot = []

    def build_prompt(self, features_text, few_shot_examples=None):
        self.received_few_shot = list(few_shot_examples or [])
        return f"PROMPT::{features_text}::N={len(self.received_few_shot)}"

    def parse(self, response):
        return 1 if "packed" in response else 0

    def parse_with_meta(self, response):
        label = 1 if "packed" in response else 0
        return {
            "label": label,
            "reason": "fake",
            "decision": "YES" if label else "NO",
            "decision_valid": True,
            "p_packed": None,
            "used_fallback": False,
        }


def _build_classifier(**kwargs):
    return llm_module.LLMClassifier(
        model_file="m.gguf",
        model_repo="repo/model",
        prompt_file="few_shot_binary.txt",
        feature_names=["entropy", "entropy_code_section"],
        n_ctx=128,
        n_threads=1,
        max_tokens=16,
        backend_factory=Backend,
        formatter_factory=Formatter,
        strategy_factory=Strategy,
        **kwargs,
    )


def test_fit_prepares_few_shot_examples():
    X = pd.DataFrame(
        {
            "entropy": [7.9, 7.2, 4.8, 4.5],
            "entropy_code_section": [7.1, 6.9, 4.4, 4.3],
        }
    )
    y = np.array([1, 1, 0, 0])

    clf = _build_classifier(few_shot_count=4)
    clf.fit(X, y)

    assert len(clf.few_shot_examples_) == 4


def test_predict_uses_selected_few_shot_examples():
    X_train = pd.DataFrame(
        {
            "entropy": [7.8, 4.2],
            "entropy_code_section": [7.0, 4.0],
        }
    )
    y_train = np.array([1, 0])
    X_test = pd.DataFrame(
        {
            "entropy": [6.5],
            "entropy_code_section": [6.1],
        }
    )

    clf = _build_classifier(few_shot_count=2)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    assert preds.tolist() == [1]
    assert "N=2" in clf.backend_.generated_prompts[0]


def test_predict_proba_uses_logprobs_when_available():
    X_train = pd.DataFrame(
        {
            "entropy": [7.8, 4.2],
            "entropy_code_section": [7.0, 4.0],
        }
    )
    y_train = np.array([1, 0])
    X_test = pd.DataFrame(
        {
            "entropy": [6.5],
            "entropy_code_section": [6.1],
        }
    )

    clf = _build_classifier(few_shot_count=2)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    assert proba.shape == (1, 2)
    assert float(proba[0, 1]) > float(proba[0, 0])


def test_fit_loads_few_shot_examples_from_path(tmp_path):
    csv_path = tmp_path / "examples.csv"
    csv_path.write_text(
        "entropy,entropy_code_section,label\n"
        "7.9,7.1,1\n"
        "7.2,6.9,1\n"
        "4.8,4.4,0\n"
        "4.5,4.3,0\n",
        encoding="utf-8",
    )

    X_train = pd.DataFrame({"entropy": [6.0], "entropy_code_section": [5.5]})
    y_train = np.array([1])

    clf = _build_classifier(few_shot_count=2, few_shot_examples_path=str(csv_path))
    clf.fit(X_train, y_train)

    assert len(clf.few_shot_examples_) == 2
    labels_in_examples = [e["label"] for e in clf.few_shot_examples_]
    assert set(labels_in_examples) == {0, 1}
