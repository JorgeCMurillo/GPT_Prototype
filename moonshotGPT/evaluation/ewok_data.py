"""Shared EWoK dataset loading utilities."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
_DEFAULT_FAST_ZIP = _PROJECT_DIR / "ewok_fast_jsonl.zip"
_DEFAULT_FAST_DIR = _PROJECT_DIR / "ewok_fast"
_DEFAULT_FULL_ZIP = _PROJECT_DIR / "ewok_full_jsonl.zip"
_DEFAULT_FULL_DIR = _PROJECT_DIR / "ewok_full_jsonl"
_DEFAULT_ZIP_PASSWORD = os.environ.get("EWOK_ZIP_PASSWORD", "ew2026")
_VALID_EWOK_VARIANTS = frozenset({"fast", "full"})


def _load_ewok_from_dir(src_dir: Path) -> pd.DataFrame:
    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in directory: {src_dir}")
    return pd.concat(
        [pd.read_json(fp, lines=True, encoding="utf-8") for fp in jsonl_files],
        ignore_index=True,
        sort=False,
    )


def _load_ewok_from_zip(zip_path: Path, password: Optional[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(name for name in zf.namelist() if name.lower().endswith(".jsonl"))
        if not members:
            raise FileNotFoundError(f"No .jsonl files found in zip: {zip_path}")

        frames = []
        for name in members:
            info = zf.getinfo(name)
            member_is_encrypted = bool(info.flag_bits & 0x1)
            pwd = None
            if member_is_encrypted:
                if not password:
                    raise RuntimeError(
                        f"Zip member '{name}' is encrypted. Set EWOK_ZIP_PASSWORD."
                    )
                pwd = password.encode("utf-8")
            try:
                with zf.open(name, "r", pwd=pwd) as f:
                    frames.append(pd.read_json(io.TextIOWrapper(f, encoding="utf-8"), lines=True))
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "password" in msg or "encrypted" in msg:
                    raise RuntimeError(
                        f"Failed to decrypt '{name}' in {zip_path}. "
                        "Set EWOK_ZIP_PASSWORD to the zip password."
                    ) from exc
                raise

    return pd.concat(frames, ignore_index=True, sort=False)


def _normalize_variant(variant: str | None) -> str:
    resolved = str(variant or os.environ.get("EWOK_VARIANT", "fast")).strip().lower()
    if resolved not in _VALID_EWOK_VARIANTS:
        valid = ", ".join(sorted(_VALID_EWOK_VARIANTS))
        raise ValueError(f"Unknown EWoK variant {resolved!r}; expected one of: {valid}")
    return resolved


def _ewok_source_candidates(variant: str) -> list[Path]:
    variant = _normalize_variant(variant)
    candidates = []
    if variant == "fast":
        env_keys = ("EWOK_FAST_SRC", "EWOK_SRC")
        defaults = (_DEFAULT_FAST_ZIP, _DEFAULT_FAST_DIR)
    else:
        env_keys = ("EWOK_FULL_SRC",)
        defaults = (_DEFAULT_FULL_ZIP, _DEFAULT_FULL_DIR)

    for env_key in env_keys:
        env_src = os.environ.get(env_key)
        if env_src:
            candidates.append(Path(env_src))
    candidates.extend(defaults)

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def load_ewok_df(variant: str | None = None) -> tuple[pd.DataFrame, Path]:
    """Load EWoK from the first available source and return `(df, source_path)`."""
    resolved_variant = _normalize_variant(variant)
    errors = []
    candidates = _ewok_source_candidates(resolved_variant)
    for src in candidates:
        try:
            if src.is_file() and src.suffix.lower() == ".zip":
                return _load_ewok_from_zip(src, _DEFAULT_ZIP_PASSWORD), src
            if src.is_dir():
                return _load_ewok_from_dir(src), src
        except Exception as exc:
            errors.append(f"{src}: {exc}")

    searched = ", ".join(str(candidate) for candidate in candidates)
    details = "; ".join(errors) if errors else "No candidate source exists."
    raise FileNotFoundError(
        f"Could not load EWoK data for variant={resolved_variant!r}. "
        f"Searched: {searched}. Details: {details}"
    )


__all__ = ["load_ewok_df"]
