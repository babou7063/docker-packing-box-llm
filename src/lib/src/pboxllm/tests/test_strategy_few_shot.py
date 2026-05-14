# -*- coding: UTF-8 -*-
from pboxllm.strategy import PromptStrategy


def test_build_prompt_injects_few_shot_examples(tmp_path):
    prompt_dir = tmp_path / "prompt-cache"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    template = "Intro\n{few_shot_examples}\nTarget:\n{features}\nAnswer."
    (prompt_dir / "few.txt").write_text(template, encoding="utf-8")

    strategy = PromptStrategy("few.txt", prompt_dir=prompt_dir, default_prompt_dir=prompt_dir)
    prompt = strategy.build_prompt(
        "entropy: 7.2",
        few_shot_examples=[
            {"features_text": "entropy: 7.9", "label": 1},
            {"features_text": "entropy: 4.2", "label": 0},
        ],
    )

    assert "Example 1" in prompt
    assert "Label: packed" in prompt
    assert "Label: not-packed" in prompt
    assert "entropy: 7.2" in prompt
    assert "Here are some examples" in prompt


def test_build_prompt_zero_shot_renders_no_examples_header(tmp_path):
    """When few_shot_examples is empty the placeholder collapses to empty string."""
    prompt_dir = tmp_path / "prompt-cache"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    template = "Intro\n{few_shot_examples}\nTarget:\n{features}\nAnswer."
    (prompt_dir / "few.txt").write_text(template, encoding="utf-8")

    strategy = PromptStrategy("few.txt", prompt_dir=prompt_dir, default_prompt_dir=prompt_dir)
    prompt = strategy.build_prompt("entropy: 5.1", few_shot_examples=[])

    assert "Here are some examples" not in prompt
    assert "No few-shot" not in prompt
    assert "entropy: 5.1" in prompt


def test_build_prompt_keeps_plain_template_compatibility(tmp_path):
    """Templates without {few_shot_examples} still work (zero-shot style)."""
    prompt_dir = tmp_path / "prompt-cache"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "zero.txt").write_text("Features:\n{features}", encoding="utf-8")

    strategy = PromptStrategy("zero.txt", prompt_dir=prompt_dir, default_prompt_dir=prompt_dir)
    prompt = strategy.build_prompt("entropy: 5.1")

    assert prompt == "Features:\nentropy: 5.1"

