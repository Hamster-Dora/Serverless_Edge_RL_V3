"""
Evaluation metrics tracking utilities for V3 experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def summarize_step_records(
    step_df: pd.DataFrame,
    policy_key: str,
    policy_label: str,
    total_reward: float,
) -> Dict[str, float | int | str]:
    """Build one-row summary metrics from step-level evaluation records."""
    if step_df.empty:
        raise RuntimeError(f"{policy_label} produced no evaluation records.")

    total_steps = len(step_df)
    valid_steps = int((step_df["start_state"] != "invalid").sum())
    invalid_steps = int((step_df["start_state"] == "invalid").sum())

    return {
        "policy": policy_label,
        "policy_key": policy_key,
        "total_requests": total_steps,
        "valid_requests": valid_steps,
        "invalid_requests": invalid_steps,
        "invalid_ratio": invalid_steps / max(total_steps, 1),
        "total_reward": float(total_reward),
        "avg_reward": float(step_df["reward"].mean()),
        "avg_latency_ms": float(step_df["latency_ms"].mean()),
        "p95_latency_ms": float(step_df["latency_ms"].quantile(0.95)),
        "p99_latency_ms": float(step_df["latency_ms"].quantile(0.99)),
        "tail_gap_ms": float(step_df["latency_ms"].quantile(0.99) - step_df["latency_ms"].mean()),
        "avg_memory_pressure": float(step_df["memory_pressure"].mean()),
        "peak_memory_pressure": float(step_df["memory_pressure"].max()),
        "final_cache_hit_rate": float(step_df["cache_hit_rate"].iloc[-1]),
        "hot_count": int((step_df["start_state"] == "hot").sum()),
        "warm_count": int((step_df["start_state"] == "warm").sum()),
        "cold_count": int((step_df["start_state"] == "cold").sum()),
        "hot_ratio": float((step_df["start_state"] == "hot").mean()),
        "warm_ratio": float((step_df["start_state"] == "warm").mean()),
        "cold_ratio": float((step_df["start_state"] == "cold").mean()),
    }


def save_step_metrics(step_df: pd.DataFrame, policy_key: str, metrics_dir: Path) -> Path:
    """Save step-level evaluation records."""
    out_path = metrics_dir / f"{policy_key}_step_metrics.csv"
    step_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def save_summary(summary_df: pd.DataFrame, metrics_dir: Path) -> Path:
    """Save summary table for all evaluated policies."""
    out_path = metrics_dir / "evaluation_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def save_text_report(summary_df: pd.DataFrame, metrics_dir: Path) -> Path:
    """Write a concise text report for quick inspection."""
    best_latency_row = summary_df.sort_values("avg_latency_ms", ascending=True).iloc[0]
    best_tail_row = summary_df.sort_values("p95_latency_ms", ascending=True).iloc[0]
    best_cache_row = summary_df.sort_values("final_cache_hit_rate", ascending=False).iloc[0]

    lines: List[str] = [
        "Serverless Edge RL V3 Evaluation Report",
        "=" * 40,
        "",
        f"Lowest average latency : {best_latency_row['policy']} ({best_latency_row['avg_latency_ms']:.2f} ms)",
        f"Lowest P95 latency     : {best_tail_row['policy']} ({best_tail_row['p95_latency_ms']:.2f} ms)",
        f"Highest cache hit rate : {best_cache_row['policy']} ({best_cache_row['final_cache_hit_rate']:.2%})",
        "",
        "Per-policy summary:",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.policy}: avg_latency={row.avg_latency_ms:.2f} ms, "
            f"p95={row.p95_latency_ms:.2f} ms, cache_hit={row.final_cache_hit_rate:.2%}, "
            f"avg_memory={row.avg_memory_pressure:.2%}, invalid={row.invalid_requests}"
        )

    out_path = metrics_dir / "evaluation_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
