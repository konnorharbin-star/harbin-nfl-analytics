"""Canonical sportsbook identity for breadth and concentration accounting."""

from __future__ import annotations

import re


def canonical_book_identity(value: object) -> str:
    """Return a stable sportsbook identity without conflating provider feed aliases."""

    raw = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    if not compact:
        return ""
    if compact.startswith("draftkings"):
        return "draftkings"
    if compact.startswith("espnbet"):
        return "espnbet"
    if compact.startswith("fanduel"):
        return "fanduel"
    if compact.startswith("betmgm"):
        return "betmgm"
    if compact.startswith("caesars"):
        return "caesars"
    if compact.startswith("fanatics"):
        return "fanatics"
    return compact
