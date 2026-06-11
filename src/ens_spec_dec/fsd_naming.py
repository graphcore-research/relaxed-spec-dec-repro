"""Canonical naming helpers for FSD-family relaxed methods."""

from __future__ import annotations

import re

RFSD_METHOD = "rfsd"
FSD_METHOD = "fsd"
LEGACY_FUZZY_METHOD = "fuzzy"
RFSD_JS_METHOD_ID = "rfsd_js"
LEGACY_FUZZY_JS_METHOD_ID = "fuzzy_js"

_LEGACY_FUZZY_JS_KEY_RE = re.compile(
    r"^fuzzy-js-fuzzy-threshold-(?P<value>.+)-d(?P<draft_length>\d+)$"
)


def canonical_relaxed_target_method(method: object) -> str:
    """Return the canonical relaxed-target method name."""

    name = str(method)
    if name == LEGACY_FUZZY_METHOD:
        return RFSD_METHOD
    return name


def canonical_method_id(method_id: object) -> str:
    """Return the canonical reporting/launch method id."""

    name = str(method_id)
    if name == LEGACY_FUZZY_JS_METHOD_ID:
        return RFSD_JS_METHOD_ID
    return name


def canonical_engine_key_id(engine_key_id: object) -> str:
    """Return the canonical engine key id for legacy FSD-family ids."""

    key = str(engine_key_id)
    match = _LEGACY_FUZZY_JS_KEY_RE.match(key)
    if match is None:
        return key
    return (
        f"rfsd-js-threshold-{match.group('value')}"
        f"-d{match.group('draft_length')}"
    )


def is_legacy_fuzzy_engine_key(engine_key_id: object) -> bool:
    """Return True for pre-rFSD fuzzy JS weekend engine keys."""

    return _LEGACY_FUZZY_JS_KEY_RE.match(str(engine_key_id)) is not None
