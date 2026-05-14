# -*- coding: UTF-8 -*-
"""Validate LLM ablation algorithm entries: feature names exist in features.yml."""
import re
from pathlib import Path

import pytest
import yaml

from pboxllm.formatter import FeatureFormatter


def _conf_dir():
    # .../src/lib/src/pboxllm/tests -> parents[4] == .../src
    return Path(__file__).resolve().parents[4] / "conf"


def _load_feature_keys():
    """Top-level feature keys from features.yml (avoid parsing !!python tags)."""
    path = _conf_dir() / "features.yml"
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0] in " \t":
            continue
        if line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k = line.split(":", 1)[0].strip()
        if k:
            keys.add(k)
    # Parameterized feature in features.yml expands at load time (see Features._load).
    if "byte_%d_after_ep" in keys:
        keys.discard("byte_%d_after_ep")
        keys.update(f"byte_{i}_after_ep" for i in range(64))
    return keys


def _load_llm_algorithm(name):
    path = _conf_dir() / "algorithms.yml"
    text = path.read_text(encoding="utf-8")
    cleaned = text.replace("!!python", "")
    data = yaml.safe_load(cleaned) or {}
    llm = data.get("LLM") or {}
    if name not in llm:
        pytest.fail(f"Algorithm {name!r} missing under LLM in algorithms.yml")
    return llm[name]


def _feature_names_for(algo_name):
    algo = _load_llm_algorithm(algo_name)
    static = (algo.get("parameters") or {}).get("static") or {}
    names = static.get("feature_names")
    assert names, f"{algo_name}: no feature_names"
    return list(names)


def _assert_features_registered(names, registry):
    missing = [n for n in names if n not in registry]
    assert not missing, f"Unknown feature names: {missing}"


def test_ablation_1_pe_layout_features_exist():
    registry = _load_feature_keys()
    names = _feature_names_for("LLM-Ablation-1-PE-Layout")
    _assert_features_registered(names, registry)


def test_ablation_1_pe_layout_formatter_runs():
    names = _feature_names_for("LLM-Ablation-1-PE-Layout")
    row = {n: 0 for n in names}
    row["entropy"] = 7.0  # not in list; ensure ignored
    fmt = FeatureFormatter(names, representation_style="flat")
    fmt._descriptions = {}
    fmt._descriptions_loaded = True
    text = fmt.format(row)
    assert len(text) > 0
    assert "vsize" in text.lower() or "section" in text.lower()


def test_ablation_2_import_surface_features_exist():
    registry = _load_feature_keys()
    names = _feature_names_for("LLM-Ablation-2-Import-Surface")
    _assert_features_registered(names, registry)


def test_ablation_2_import_surface_formatter_runs():
    names = _feature_names_for("LLM-Ablation-2-Import-Surface")
    row = {n: 1 for n in names}
    fmt = FeatureFormatter(names, representation_style="flat")
    fmt._descriptions = {}
    fmt._descriptions_loaded = True
    text = fmt.format(row)
    assert len(text) > 0
    assert "import" in text.lower() or "dll" in text.lower() or "iat" in text.lower()


def test_ablation_3_stats_compact_features_exist():
    registry = _load_feature_keys()
    names = _feature_names_for("LLM-Ablation-3-Stats-Compact")
    _assert_features_registered(names, registry)


def test_ablation_3_stats_compact_formatter_runs():
    names = _feature_names_for("LLM-Ablation-3-Stats-Compact")
    row = {n: 7.0 if "entropy" in n else False for n in names}
    fmt = FeatureFormatter(names, representation_style="flat")
    fmt._descriptions = {}
    fmt._descriptions_loaded = True
    text = fmt.format(row)
    assert len(text) > 0
    assert "entropy" in text.lower()


def test_ablation_4_strings_summary_features_exist():
    registry = _load_feature_keys()
    names = _feature_names_for("LLM-Ablation-4-Strings-Summary")
    _assert_features_registered(names, registry)


def test_ablation_4_strings_summary_formatter_runs():
    names = _feature_names_for("LLM-Ablation-4-Strings-Summary")
    row = {n: 0 for n in names}
    fmt = FeatureFormatter(names, representation_style="strings_summary")
    fmt._descriptions = {}
    fmt._descriptions_loaded = True
    text = fmt.format(row)
    assert text.startswith("STRINGS_SUMMARY")
    assert "string" in text.lower()


def test_ablation_5_ep_byte_window_features_exist():
    registry = _load_feature_keys()
    names = _feature_names_for("LLM-Ablation-5-EP-Byte-Window")
    _assert_features_registered(names, registry)
    assert len(names) == 64


def test_ablation_5_ep_byte_window_formatter_runs():
    names = _feature_names_for("LLM-Ablation-5-EP-Byte-Window")
    row = {n: i % 256 for i, n in enumerate(names)}
    fmt = FeatureFormatter(names, representation_style="flat")
    fmt._descriptions = {}
    fmt._descriptions_loaded = True
    text = fmt.format(row)
    assert len(text) > 0
    assert "byte" in text.lower() or "ep" in text.lower()
