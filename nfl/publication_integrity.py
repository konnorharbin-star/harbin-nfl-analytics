"""Deterministic multi-page publication manifest and integrity checks.

The manifest is a checksum ledger, NOT a statement that an advertised quote
is executable. It protects the public board, research labels and betting CSV
against partial commits, cached PNGs and stale docs copies.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 1
DIGEST_NAME = "sha256"
PAGE_SIZE = 14


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("canonical generated_at timestamp is missing")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid canonical generated_at timestamp") from exc
    if stamp.tzinfo is None:
        raise ValueError("canonical generated_at must include UTC offset")
    return stamp.astimezone(UTC)


def _run_tag(timestamp: datetime) -> str:
    local = timestamp.astimezone(ZoneInfo("America/Chicago"))
    return local.strftime("%Y%m%d_%H%M%S_CT")


def _display(timestamp: datetime) -> str:
    local = timestamp.astimezone(ZoneInfo("America/Chicago"))
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.strftime('%b')} {local.day}, {local.year} · {clock} CT"


def _source_path(path: object, root: Path) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("empty publication file path")
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"publication path escapes repository: {path}")
    return source


def _read_json(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"expected JSON object at {path}")
    return source


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_publication_manifest(
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> dict[str, object]:
    """Build an exact checksum list tied to the live run, not historical files."""
    outputs = Path(output_dir).resolve()
    docs = Path(docs_dir).resolve()
    root = outputs.parent
    if docs.parent != root:
        raise ValueError("outputs and docs must belong to the same repository root")
    report = _read_json(outputs / "current_model.json")
    meta = report.get("meta")
    publication = report.get("publication")
    if not isinstance(meta, dict) or not isinstance(publication, dict):
        raise ValueError("canonical model metadata/publication missing")
    timestamp = _timestamp(meta.get("generated_at"))
    report_at = _timestamp(report.get("generated_at"))
    if report_at < timestamp:
        raise ValueError("report was created before its canonical run metadata")
    try:
        season = int(meta["season"])
        week = int(meta["week"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("canonical season/week missing") from exc
    if not 1 <= week <= 22:
        raise ValueError("canonical NFL week outside supported range")
    release = str(report.get("release_state") or "")
    if release not in {"RESEARCH", "PRODUCTION", "BLOCKED"}:
        raise ValueError("canonical release state missing or invalid")

    stable = publication.get("png_pages")
    fresh = publication.get("cache_safe_png_pages")
    if (
        not isinstance(stable, list) or not stable
        or not isinstance(fresh, list) or len(stable) != len(fresh)
    ):
        raise ValueError("stable and fresh PNG page sets must be complete and equal")
    if len({str(x) for x in stable}) != len(stable):
        raise ValueError("duplicate stable PNG page paths")
    if len({str(x) for x in fresh}) != len(fresh):
        raise ValueError("duplicate cache-safe PNG page paths")
    tag = str(publication.get("run_tag") or "")
    if tag and tag != _run_tag(timestamp):
        raise ValueError("cache-safe PNG run tag does not match model timestamp")
    board_games = publication.get("board_games")
    if board_games is not None:
        try:
            count = int(board_games)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid publication board game count") from exc
        if count < 0 or len(stable) != max(1, math.ceil(count / PAGE_SIZE)):
            raise ValueError("published PNG page count disagrees with the board")
    paths = [
        outputs / "current_model.json",
        outputs / "README.md",
        outputs / "current_predictions.csv",
        outputs / "suggested_bets.csv",
        outputs / "quant_recommendations.csv",
        outputs / "quant_card.html",
        outputs / "latest.png",
        docs / "latest.json",
        docs / "latest.csv",
        docs / "latest.png",
        docs / "index.html",
        docs / "quant.html",
        docs / "audit_snapshot.json",
    ]
    for page, (stable_value, fresh_value) in enumerate(
        zip(stable, fresh, strict=True), start=1
    ):
        stable_path = _source_path(stable_value, root)
        fresh_path = _source_path(fresh_value, root)
        if stable_path.parent != outputs or fresh_path.parent != outputs:
            raise ValueError("PNG pages must be stored inside the output directory")
        expected_stable = f"nfl_week_{week}_page{page}.png"
        if stable_path.name != expected_stable:
            raise ValueError(f"stable page name is wrong for page {page}")
        if tag:
            expected_fresh = f"nfl_week_{week}_run_{tag}_page{page}.png"
            if fresh_path.name != expected_fresh:
                raise ValueError(f"cache-safe page {page} uses the wrong run timestamp")
            if fresh_path == stable_path:
                raise ValueError("cache-safe pages must be distinct from stable pages")
        paths.extend([stable_path, fresh_path])
    if len(set(paths)) != len(paths) and tag:
        raise ValueError("publication files contain duplicate path references")

    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"required publication file is empty or missing: {path.name}")

    if _digest(outputs / "latest.png") != _digest(docs / "latest.png"):
        raise ValueError("public latest.png differs from model latest.png")
    if _digest(outputs / "latest.png") != _digest(_source_path(stable[0], root)):
        raise ValueError("latest.png is stale versus stable page 1")
    if _digest(docs / "latest.json") != _digest(outputs / "current_model.json"):
        raise ValueError("public latest.json differs from canonical current_model.json")
    if _digest(docs / "latest.csv") != _digest(outputs / "current_predictions.csv"):
        raise ValueError("public latest.csv differs from current_predictions.csv")
    if _digest(docs / "quant.html") != _digest(outputs / "quant_card.html"):
        raise ValueError("public quant card differs from current output")
    html_path = _source_path(publication.get("html"), root)
    if not html_path.is_file() or _digest(html_path) != _digest(docs / "index.html"):
        raise ValueError("public HTML board differs from current weekly HTML")
    paths.append(html_path)

    for page, (stable_value, fresh_value) in enumerate(
        zip(stable, fresh, strict=True), start=1
    ):
        stable_path = _source_path(stable_value, root)
        fresh_path = _source_path(fresh_value, root)
        if _digest(stable_path) != _digest(fresh_path):
            raise ValueError(f"cache-safe PNG page {page} is stale")
        if tag and not stable_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"PNG page {page} has an invalid image signature")

    readme = (outputs / "README.md").read_text(encoding="utf-8")
    required = [
        f"**Season / Week:** {season} / {week}",
        f"**Updated:** {_display(timestamp)}",
        f"**Release state:** {release}",
        *[f"]({Path(str(x)).name})" for x in stable],
        *[f"]({Path(str(x)).name})" for x in fresh],
    ]
    if any(value not in readme for value in required):
        raise ValueError("README timestamp, release, week or current PNG links are stale")

    fingerprints = []
    for path in sorted(set(paths)):
        fingerprints.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp.isoformat(),
        "report_generated_at": report_at.isoformat(),
        "season": season,
        "week": week,
        "release_state": release,
        "run_tag": tag or None,
        "board_games": board_games,
        "page_count": len(stable),
        "file_count": len(fingerprints),
        "files": fingerprints,
        "status": "PASS",
        "scope": (
            "Exactly one canonical run: model, forecasts, suggestions, "
            "all PNG pages, live README and public docs; no claim of bet fills."
        ),
    }


def write_publication_manifest(
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> dict[str, object]:
    """Write identical manifests only after the complete publication exists."""
    payload = build_publication_manifest(output_dir=output_dir, docs_dir=docs_dir)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (Path(output_dir) / "publication_manifest.json").write_text(
        data, encoding="utf-8"
    )
    (Path(docs_dir) / "publication_manifest.json").write_text(
        data, encoding="utf-8"
    )
    return payload


def validate_publication_manifest(
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> dict[str, object]:
    """Detect partial, obsolete and cross-run public files after publishing."""
    outputs = Path(output_dir)
    docs = Path(docs_dir)
    try:
        actual = build_publication_manifest(output_dir=outputs, docs_dir=docs)
        output_manifest = _read_json(outputs / "publication_manifest.json")
        public_manifest = _read_json(docs / "publication_manifest.json")
        if output_manifest != public_manifest:
            raise ValueError("public manifest differs from model manifest")
        if output_manifest != actual:
            diffs = [
                item["path"] for item in actual["files"]
                if not any(
                    prior == item for prior in output_manifest.get("files", [])
                )
            ]
            raise ValueError(
                "publication manifest does not match current files: "
                + ", ".join(diffs[:8])
            )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": str(exc), "file_count": 0}
    return {
        "status": "PASS",
        "reason": "all canonical files and every PNG page match this run",
        "file_count": actual["file_count"],
        "run_tag": actual["run_tag"],
        "generated_at": actual["generated_at"],
    }
