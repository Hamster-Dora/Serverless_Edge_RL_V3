"""
Fixed vocabulary utilities for stable multi-day PPO experiments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from utils.trace_parser import AzureTraceParser


def parse_csv_items(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Argument must contain at least one non-empty item.")
    return items


def build_vocab_from_frames(frames: Sequence[pd.DataFrame]) -> dict[str, list[str]]:
    function_names: set[str] = set()
    layer_names: set[str] = set()

    for frame in frames:
        if frame.empty:
            continue
        function_names.update(frame["HashFunction"].astype(str).unique().tolist())
        layer_names.update(frame["Layer_ID"].astype(str).unique().tolist())

    if not function_names or not layer_names:
        raise ValueError("Cannot build vocabulary from empty frames.")

    return {
        "function_names": sorted(function_names),
        "layer_names": sorted(layer_names),
    }


def _build_full_day_vocab_fast(day: str, data_dir: str | None = None) -> dict[str, set[str]]:
    """
    Build a day-level vocabulary without expanding the invocation table into a melted request stream.
    This is much cheaper than `load_day_data()` and is suitable for `rows_per_day == 0`.
    """
    parser = AzureTraceParser(data_dir=data_dir)
    invocations_path = os.path.join(
        parser.data_dir,
        f"invocations_per_function_md.anon.{day}.csv",
    )
    invocations = pd.read_csv(
        invocations_path,
        usecols=["HashApp", "HashFunction"],
        dtype={"HashApp": "string", "HashFunction": "string"},
    )
    hash_apps = invocations["HashApp"].dropna().astype(str)
    hash_functions = invocations["HashFunction"].dropna().astype(str)
    return {
        "function_names": set(hash_functions.unique().tolist()),
        "layer_names": set(hash_apps.str[:4].unique().tolist()),
    }


def build_vocab_from_days(
    days: Sequence[str],
    rows_per_day: int = 0,
    data_dir: str | None = None,
) -> dict[str, list[str]]:
    function_names: set[str] = set()
    layer_names: set[str] = set()
    parser = AzureTraceParser(data_dir=data_dir) if rows_per_day and rows_per_day > 0 else None

    for day in days:
        if rows_per_day and rows_per_day > 0:
            frame = parser.load_day_data(day)
            frame = frame.iloc[:rows_per_day].copy()
            function_names.update(frame["HashFunction"].astype(str).unique().tolist())
            layer_names.update(frame["Layer_ID"].astype(str).unique().tolist())
        else:
            vocab_chunk = _build_full_day_vocab_fast(day=day, data_dir=data_dir)
            function_names.update(vocab_chunk["function_names"])
            layer_names.update(vocab_chunk["layer_names"])

    if not function_names or not layer_names:
        raise ValueError("Cannot build vocabulary from empty day set.")

    return {
        "function_names": sorted(function_names),
        "layer_names": sorted(layer_names),
    }


def vocab_sidecar_path(model_path: str | Path) -> Path:
    candidate = Path(model_path)
    base = candidate.with_suffix("") if candidate.suffix == ".zip" else candidate
    return base.parent / f"{base.name}_vocab.json"


def save_vocab(
    vocab: dict[str, list[str]],
    target_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    path = Path(target_path)
    payload = {
        "function_names": vocab["function_names"],
        "layer_names": vocab["layer_names"],
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_vocab(target_path: str | Path) -> dict[str, list[str]] | None:
    path = Path(target_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "function_names": list(payload["function_names"]),
        "layer_names": list(payload["layer_names"]),
    }
