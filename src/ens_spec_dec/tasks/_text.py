"""Tiny shared text helpers for generation-based benchmark adapters."""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_qwen_thinking(text: str) -> str:
    # Cactus strips Qwen3 thinking traces before filters/scoring. Keep that
    # scorer-side behavior independent of whether a run disables thinking.
    # https://github.com/ScalingIntelligence/Cactus/blob/main/src/cactus/eval/utils.py
    without_blocks = _THINK_BLOCK_RE.sub("", text)
    return _UNCLOSED_THINK_RE.sub("", without_blocks).strip()


def text_after_qwen_thinking(text: str) -> str | None:
    # Generated Qwen text usually starts after the prompt-owned opening
    # `<think>`, so the closing marker is the reliable answer-phase boundary.
    marker = "</think>"
    if marker not in text:
        return None
    return text.split(marker, 1)[1].strip()


__all__ = ["strip_qwen_thinking", "text_after_qwen_thinking"]
