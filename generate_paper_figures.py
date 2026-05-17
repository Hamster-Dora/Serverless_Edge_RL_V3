"""
Generate paper-ready figures from existing phase-6 experiment outputs.

This script is intentionally read-only with respect to experiment data:
it only reads CSV summaries / step metrics and writes figures plus
one manifest CSV under ``results/paper_figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.plotter import plot_startup_mix, plot_summary_bars, plot_time_series


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PAPER_DIR = RESULTS_DIR / "paper_figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_ROOT = RESULTS_DIR / "comparison" / "20260503_200511_phase6_final_comparison_holdout_seed42"
ABLATION_ROOT = RESULTS_DIR / "ablation" / "20260504_124317_phase6_final_ablation_quick_seed42"


def add_manifest(rows: list[dict[str, str]], name: str, path: Path, experiment: str, description: str) -> None:
    rows.append(
        {
            "figure_name": name,
            "path": str(path.relative_to(ROOT)),
            "experiment": experiment,
            "description": description,
        }
    )


def generate_day_level_base_plots(manifest_rows: list[dict[str, str]]) -> None:
    comparison_runs = ["d13_seed42", "d14_seed42"]
    for run_name in comparison_runs:
        metrics_dir = COMPARISON_ROOT / run_name / "metrics_csv"
        plots_dir = COMPARISON_ROOT / run_name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.read_csv(metrics_dir / "evaluation_summary.csv")
        step_frames = [
            pd.read_csv(metrics_dir / "ppo_step_metrics.csv"),
            pd.read_csv(metrics_dir / "lru_step_metrics.csv"),
            pd.read_csv(metrics_dir / "greedy_step_metrics.csv"),
            pd.read_csv(metrics_dir / "random_step_metrics.csv"),
        ]

        plot_summary_bars(summary_df, plots_dir)
        plot_startup_mix(summary_df, plots_dir)
        plot_time_series(step_frames, plots_dir)

        for name in [
            "policy_summary.png",
            "startup_mix.png",
            "latency_trajectory.png",
            "memory_pressure_trajectory.png",
        ]:
            add_manifest(
                manifest_rows,
                f"{run_name}_{name}",
                plots_dir / name,
                "phase6_comparison_day_level",
                f"Day-level comparison figure for {run_name}.",
            )

    ablation_runs = ["d13_seed42", "d14_seed42"]
    for run_name in ablation_runs:
        metrics_dir = ABLATION_ROOT / "evaluations" / run_name / "metrics_csv"
        plots_dir = ABLATION_ROOT / "evaluations" / run_name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.read_csv(metrics_dir / "evaluation_summary.csv")
        step_frames = [
            pd.read_csv(metrics_dir / "full_v3_step_metrics.csv"),
            pd.read_csv(metrics_dir / "fixed_beta_step_metrics.csv"),
            pd.read_csv(metrics_dir / "no_layer_sharing_step_metrics.csv"),
        ]

        plot_summary_bars(summary_df, plots_dir)
        plot_startup_mix(summary_df, plots_dir)
        plot_time_series(step_frames, plots_dir)

        for name in [
            "policy_summary.png",
            "startup_mix.png",
            "latency_trajectory.png",
            "memory_pressure_trajectory.png",
        ]:
            add_manifest(
                manifest_rows,
                f"{run_name}_{name}",
                plots_dir / name,
                "phase6_ablation_day_level",
                f"Day-level ablation figure for {run_name}.",
            )


def generate_comparison_overview(manifest_rows: list[dict[str, str]]) -> None:
    comp = pd.read_csv(COMPARISON_ROOT / "grouped_policy_summary.csv")
    labels = comp["policy"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(x, comp["avg_latency_ms_mean"], color="#4C78A8")
    axes[0, 0].set_title("Phase6 Comparison: Average Latency")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("ms")

    axes[0, 1].bar(x, comp["tail_gap_ms_mean"], color="#F58518")
    axes[0, 1].set_title("Phase6 Comparison: Tail Gap")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("ms")

    axes[1, 0].bar(x, comp["final_cache_hit_rate_mean"], color="#54A24B")
    axes[1, 0].set_title("Phase6 Comparison: Cache Hit Rate")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].bar(x, comp["invalid_ratio_mean"], color="#E45756")
    axes[1, 1].set_title("Phase6 Comparison: Invalid Ratio")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_comparison_overview.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_comparison",
        "Main comparison overview: avg latency, tail gap, cache hit rate, invalid ratio.",
    )


def generate_ablation_overview(manifest_rows: list[dict[str, str]]) -> None:
    abl = pd.read_csv(ABLATION_ROOT / "grouped_ablation_summary.csv")
    abl_mean = abl.groupby("policy", as_index=False).mean(numeric_only=True)
    labels = abl_mean["policy"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(x, abl_mean["avg_latency_ms_mean"], color="#4C78A8")
    axes[0, 0].set_title("Phase6 Ablation: Average Latency")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("ms")

    axes[0, 1].bar(x, abl_mean["p99_latency_ms_mean"], color="#F58518")
    axes[0, 1].set_title("Phase6 Ablation: P99 Latency")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("ms")

    axes[1, 0].bar(x, abl_mean["tail_gap_ms_mean"], color="#54A24B")
    axes[1, 0].set_title("Phase6 Ablation: Tail Gap")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("ms")

    axes[1, 1].bar(x, abl_mean["final_cache_hit_rate_mean"], color="#E45756")
    axes[1, 1].set_title("Phase6 Ablation: Cache Hit Rate")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_ablation_overview.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_ablation",
        "Main ablation overview: avg latency, p99, tail gap, cache hit rate.",
    )


def generate_comparison_by_day(manifest_rows: list[dict[str, str]]) -> None:
    frames = []
    for day in ["d13", "d14"]:
        path = COMPARISON_ROOT / f"{day}_seed42" / "metrics_csv" / "evaluation_summary.csv"
        frames.append(pd.read_csv(path))
    all_df = pd.concat(frames, ignore_index=True)

    metrics = [
        ("avg_latency_ms", "Average Latency (ms)"),
        ("tail_gap_ms", "Tail Gap (ms)"),
        ("final_cache_hit_rate", "Cache Hit Rate"),
        ("invalid_ratio", "Invalid Ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        pivot = all_df.pivot(index="policy", columns="day", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="day")

    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_comparison_by_day.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_comparison",
        "Comparison metrics split by hold-out day d13/d14.",
    )


def generate_startup_mix_aggregate(manifest_rows: list[dict[str, str]]) -> None:
    comp = pd.read_csv(COMPARISON_ROOT / "grouped_policy_summary.csv")
    labels = comp["policy"].tolist()
    x = np.arange(len(labels))

    hot = comp["hot_ratio_mean"].to_numpy()
    warm = comp["warm_ratio_mean"].to_numpy()
    cold = comp["cold_ratio_mean"].to_numpy()
    invalid = comp["invalid_ratio_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, hot, label="Hot")
    ax.bar(x, warm, bottom=hot, label="Warm")
    ax.bar(x, cold, bottom=hot + warm, label="Cold")
    ax.bar(x, invalid, bottom=hot + warm + cold, label="Invalid")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("Phase6 Comparison Startup State Composition")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_comparison_startup_mix.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_comparison",
        "Stacked startup-state composition for final comparison.",
    )

    abl = pd.read_csv(ABLATION_ROOT / "grouped_ablation_summary.csv")
    abl_mean = abl.groupby("policy", as_index=False).mean(numeric_only=True)
    labels = abl_mean["policy"].tolist()
    x = np.arange(len(labels))

    hot = abl_mean["hot_ratio_mean"].to_numpy()
    warm = abl_mean["warm_ratio_mean"].to_numpy()
    cold = abl_mean["cold_ratio_mean"].to_numpy()
    invalid = abl_mean["invalid_ratio_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, hot, label="Hot")
    ax.bar(x, warm, bottom=hot, label="Warm")
    ax.bar(x, cold, bottom=hot + warm, label="Cold")
    ax.bar(x, invalid, bottom=hot + warm + cold, label="Invalid")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("Phase6 Ablation Startup State Composition")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_ablation_startup_mix.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_ablation",
        "Stacked startup-state composition for final ablation.",
    )


def generate_latency_cdf(manifest_rows: list[dict[str, str]]) -> None:
    cdf_dpi = 320
    comparison_styles: list[tuple[str, str, str]] = [
        ("PPO", "#1f77b4", "-"),    # solid
        ("LRU", "#ff7f0e", "--"),   # dashed
        ("Greedy", "#2ca02c", ":"),  # dotted
        ("Random", "#d62728", "-."),  # dash-dot
    ]

    metrics_dir = COMPARISON_ROOT / "d13_seed42" / "metrics_csv"
    files = {
        "PPO": "ppo_step_metrics.csv",
        "LRU": "lru_step_metrics.csv",
        "Greedy": "greedy_step_metrics.csv",
        "Random": "random_step_metrics.csv",
    }

    comparison_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lw = 2.4
    for label, color, ls in comparison_styles:
        name = files[label]
        df = pd.read_csv(metrics_dir / name, usecols=["latency_ms"])
        vals = np.sort(df["latency_ms"].to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        comparison_curves[label] = (vals, y)

    all_vals = np.concatenate([vals for vals, _ in comparison_curves.values()])
    x_min = float(all_vals.min())
    x_max = float(all_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (focus_label, focus_color, focus_ls) in zip(axes, comparison_styles):
        for label, color, ls in comparison_styles:
            vals, y = comparison_curves[label]
            if label == focus_label:
                ax.plot(vals, y, color=color, linestyle=ls, linewidth=2.8, label=label, zorder=3)
            else:
                ax.plot(vals, y, color=color, linestyle=ls, linewidth=1.2, alpha=0.22, zorder=1)

        ax.set_title(f"{focus_label} (highlighted)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.92, fontsize=10)

    fig.suptitle("Latency CDF on d13 (Phase6 Comparison)", fontsize=16)
    fig.supxlabel("latency (ms)")
    fig.supylabel("CDF")
    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_comparison_d13_latency_cdf.png"
    fig.savefig(out_path, dpi=cdf_dpi, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_comparison",
        "Latency CDF on d13 for PPO vs heuristic baselines.",
    )

    ablation_styles: list[tuple[str, str, str]] = [
        ("FullV3", "#1f77b4", "-"),
        ("FixedBeta", "#ff7f0e", "--"),
        ("NoLayerSharing", "#d62728", ":"),
    ]

    metrics_dir = ABLATION_ROOT / "evaluations" / "d13_seed42" / "metrics_csv"
    files = {
        "FullV3": "full_v3_step_metrics.csv",
        "FixedBeta": "fixed_beta_step_metrics.csv",
        "NoLayerSharing": "no_layer_sharing_step_metrics.csv",
    }

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, color, ls in ablation_styles:
        name = files[label]
        df = pd.read_csv(metrics_dir / name, usecols=["latency_ms"])
        vals = np.sort(df["latency_ms"].to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_title("Latency CDF on d13 (Phase6 Ablation)")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(framealpha=0.92, fontsize=11)
    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_ablation_d13_latency_cdf.png"
    fig.savefig(out_path, dpi=cdf_dpi, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_ablation",
        "Latency CDF on d13 for FullV3 / FixedBeta / NoLayerSharing.",
    )


def generate_tail_latency_focus(manifest_rows: list[dict[str, str]]) -> None:
    comp = pd.read_csv(COMPARISON_ROOT / "grouped_policy_summary.csv")
    labels = comp["policy"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("p95_latency_ms_mean", "P95 Latency"),
        ("p99_latency_ms_mean", "P99 Latency"),
        ("tail_gap_ms_mean", "Tail Gap"),
    ]
    colors = ["#4C78A8", "#F58518", "#E45756"]
    for ax, (metric, title), color in zip(axes, metrics, colors):
        ax.bar(x, comp[metric], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylabel("ms")
    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_comparison_tail_focus.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_comparison_tail_analysis",
        "Tail-latency专题图：comparison 的 P95 / P99 / tail gap.",
    )

    abl = pd.read_csv(ABLATION_ROOT / "grouped_ablation_summary.csv")
    abl_mean = abl.groupby("policy", as_index=False).mean(numeric_only=True)
    labels = abl_mean["policy"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title), color in zip(axes, metrics, colors):
        ax.bar(x, abl_mean[metric], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylabel("ms")
    fig.tight_layout()
    out_path = PAPER_DIR / "phase6_ablation_tail_focus.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_ablation_tail_analysis",
        "Tail-latency专题图：ablation 的 P95 / P99 / tail gap.",
    )


def generate_cache_memory_joint(manifest_rows: list[dict[str, str]]) -> None:
    comp = pd.read_csv(COMPARISON_ROOT / "grouped_policy_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        comp["avg_memory_pressure_mean"],
        comp["final_cache_hit_rate_mean"],
        s=np.clip(comp["tail_gap_ms_mean"], 10, None) * 8,
        c=comp["avg_latency_ms_mean"],
        cmap="viridis",
        alpha=0.85,
    )
    for _, row in comp.iterrows():
        ax.annotate(row["policy"], (row["avg_memory_pressure_mean"], row["final_cache_hit_rate_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Phase6 Comparison: Cache Hit Rate vs Memory Pressure")
    ax.set_xlabel("Average Memory Pressure")
    ax.set_ylabel("Final Cache Hit Rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path = PAPER_DIR / "phase6_comparison_cache_memory_joint.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_comparison_joint_analysis",
        "联合图：comparison 的缓存命中率 vs 内存压力（点大小近似 tail gap）。",
    )

    abl = pd.read_csv(ABLATION_ROOT / "grouped_ablation_summary.csv")
    abl_mean = abl.groupby("policy", as_index=False).mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        abl_mean["avg_memory_pressure_mean"],
        abl_mean["final_cache_hit_rate_mean"],
        s=np.clip(abl_mean["tail_gap_ms_mean"], 10, None) / 50,
        c=abl_mean["avg_latency_ms_mean"],
        cmap="plasma",
        alpha=0.85,
    )
    for _, row in abl_mean.iterrows():
        ax.annotate(row["policy"], (row["avg_memory_pressure_mean"], row["final_cache_hit_rate_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Phase6 Ablation: Cache Hit Rate vs Memory Pressure")
    ax.set_xlabel("Average Memory Pressure")
    ax.set_ylabel("Final Cache Hit Rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path = PAPER_DIR / "phase6_ablation_cache_memory_joint.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    add_manifest(
        manifest_rows,
        out_path.name,
        out_path,
        "phase6_ablation_joint_analysis",
        "联合图：ablation 的缓存命中率 vs 内存压力（点大小近似 tail gap）。",
    )


def save_manifest(manifest_rows: list[dict[str, str]]) -> Path:
    manifest_path = PAPER_DIR / "figure_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def main() -> None:
    manifest_rows: list[dict[str, str]] = []

    generate_day_level_base_plots(manifest_rows)
    generate_comparison_overview(manifest_rows)
    generate_ablation_overview(manifest_rows)
    generate_comparison_by_day(manifest_rows)
    generate_startup_mix_aggregate(manifest_rows)
    generate_latency_cdf(manifest_rows)
    generate_tail_latency_focus(manifest_rows)
    generate_cache_memory_joint(manifest_rows)
    manifest_path = save_manifest(manifest_rows)

    print(f"Generated {len(manifest_rows)} figures.")
    print(f"Manifest: {manifest_path}")
    print(f"Paper figures directory: {PAPER_DIR}")


if __name__ == "__main__":
    main()
