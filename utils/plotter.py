"""
Plotting utilities for V3 evaluation and experiment scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_summary_bars(summary_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Plot four core bar charts required by the V3 comparison stage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = summary_df["policy"].tolist()
    x = np.arange(len(labels))

    axes[0, 0].bar(x, summary_df["avg_latency_ms"], color="#4C78A8")
    axes[0, 0].set_title("Average Latency")
    axes[0, 0].set_ylabel("ms")
    axes[0, 0].set_xticks(x, labels)

    axes[0, 1].bar(x, summary_df["p95_latency_ms"], color="#F58518")
    axes[0, 1].set_title("P95 Latency")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_xticks(x, labels)

    axes[1, 0].bar(x, summary_df["final_cache_hit_rate"], color="#54A24B")
    axes[1, 0].set_title("Cache Hit Rate")
    axes[1, 0].set_ylabel("ratio")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].bar(x, summary_df["avg_memory_pressure"], color="#E45756")
    axes[1, 1].set_title("Average Memory Pressure")
    axes[1, 1].set_ylabel("ratio")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(0.0, 1.0)

    for ax in axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("V3 Policy Comparison Summary", fontsize=14)
    fig.tight_layout()

    out_path = plots_dir / "policy_summary.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_startup_mix(summary_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Plot hot / warm / cold / invalid startup composition."""
    labels = summary_df["policy"].tolist()
    x = np.arange(len(labels))

    hot = summary_df["hot_ratio"].to_numpy(dtype=np.float32)
    warm = summary_df["warm_ratio"].to_numpy(dtype=np.float32)
    cold = summary_df["cold_ratio"].to_numpy(dtype=np.float32)
    invalid = summary_df["invalid_ratio"].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, hot, label="Hot", color="#54A24B")
    ax.bar(x, warm, bottom=hot, label="Warm", color="#4C78A8")
    ax.bar(x, cold, bottom=hot + warm, label="Cold", color="#F58518")
    ax.bar(x, invalid, bottom=hot + warm + cold, label="Invalid", color="#E45756")

    ax.set_title("Startup State Composition")
    ax.set_ylabel("ratio")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = plots_dir / "startup_mix.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_time_series(step_frames: List[pd.DataFrame], plots_dir: Path) -> Tuple[Path, Path]:
    """Plot latency and memory-pressure trajectories over the request stream."""
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    for frame in step_frames:
        ax1.plot(frame["step"], frame["avg_latency_ms"], label=frame["policy"].iloc[0], linewidth=1.6)
    ax1.set_title("Average Latency Trajectory")
    ax1.set_xlabel("request step")
    ax1.set_ylabel("ms")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    latency_path = plots_dir / "latency_trajectory.png"
    fig1.savefig(latency_path, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for frame in step_frames:
        ax2.plot(frame["step"], frame["memory_pressure"], label=frame["policy"].iloc[0], linewidth=1.6)
    ax2.set_title("Memory Pressure Trajectory")
    ax2.set_xlabel("request step")
    ax2.set_ylabel("ratio")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    memory_path = plots_dir / "memory_pressure_trajectory.png"
    fig2.savefig(memory_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    return latency_path, memory_path
