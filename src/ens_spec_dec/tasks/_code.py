"""Shared helpers for code-generation benchmark adapters."""

from __future__ import annotations

from ._text import strip_qwen_thinking


def extract_python_code(text: str) -> str:
    """Return the official LCB final fenced block after Qwen thinking text."""

    text = strip_qwen_thinking(text).strip()

    # Source-grounded: LiveCodeBench extracts the code between the final two
    # fence lines for normal chat/instruct model styles.
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/utils/extraction_utils.py
    lines = text.split("\n")
    fence_lines = [index for index, line in enumerate(lines) if "```" in line]
    if len(fence_lines) < 2:
        return ""

    start, end = fence_lines[-2], fence_lines[-1]
    return "\n".join(lines[start + 1 : end]).strip()


__all__ = ["extract_python_code"]
