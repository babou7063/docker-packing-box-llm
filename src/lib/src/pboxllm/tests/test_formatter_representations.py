# -*- coding: UTF-8 -*-
from pboxllm.formatter import FeatureFormatter


def _fake_row():
    return {
        "entropy": 7.1234,
        "number_dll_imported": 12,
        "number_strings": 340,
        "number_urls": 3,
        "sample_list": ["a", "b", "c"],
    }

def _prepare_formatter(formatter):
    # Avoid importing full pbox registry in unit tests.
    formatter._descriptions = {}
    formatter._descriptions_loaded = True
    return formatter


def test_format_flat_default():
    f = _prepare_formatter(FeatureFormatter(["entropy", "number_dll_imported"], representation_style="flat"))
    text = f.format(_fake_row())
    assert "7.1234" in text
    assert "12" in text


def test_format_structured_semantic_pe():
    f = _prepare_formatter(FeatureFormatter(
        ["entropy", "number_dll_imported", "number_strings"],
        representation_style="structured_semantic_pe",
    ))
    text = f.format(_fake_row())
    assert "PE_STATIC_REPORT {" in text
    assert "entropy_and_entrypoint" in text
    assert "imports_and_apis" in text


def test_format_compact_pe_structural():
    f = _prepare_formatter(FeatureFormatter(["entropy", "number_dll_imported"], representation_style="compact_pe_structural"))
    text = f.format(_fake_row())
    assert text.startswith("COMPACT_PE_BUNDLE")


def test_format_strings_summary():
    f = _prepare_formatter(FeatureFormatter(["number_strings", "number_urls"], representation_style="strings_summary"))
    text = f.format(_fake_row())
    assert text.startswith("STRINGS_SUMMARY")

