"""
自动化对比实验入口。

职责定位：
1. 批量运行 PPO / LRU / Greedy / Random 的统一评估
2. 支持多 day、多 seed 的对比实验，便于从当前 d01 测试平滑过渡到论文终稿实验
3. 将每次运行的结果归档到独立目录，并额外生成跨运行聚合汇总表

说明：
- 单次评估与绘图逻辑复用 `main_evaluate.py`
- 本脚本负责“实验编排”和“结果归档”
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_evaluate import (  # noqa: E402
    build_env,
    evaluate_policy,
    get_requested_policies,
    load_policies,
    resolve_model_path,
    validate_model_env_compatibility,
)
from utils.metrics_tracker import save_step_metrics, save_summary, save_text_report  # noqa: E402
from utils.plotter import plot_startup_mix, plot_summary_bars, plot_time_series  # noqa: E402
from utils.trace_parser import AzureTraceParser  # noqa: E402
from utils.vocab import build_vocab_from_days, load_vocab, parse_csv_items as parse_vocab_items, vocab_sidecar_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Run batched PPO vs heuristic comparisons.")
    p.add_argument(
        "--days",
        type=str,
        default="d01",
        help="Comma-separated Azure trace days, e.g. d01,d02,d03",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=20_000,
        help="Max rows from parsed trace per day (0 = use all).",
    )
    p.add_argument("--nodes", type=int, default=3, help="Number of edge nodes")
    p.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds, e.g. 42,43,44",
    )
    p.add_argument(
        "--policies",
        type=str,
        default="ppo,lru,greedy,random",
        help="Comma-separated policies to evaluate: ppo,lru,greedy,random",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "models" / "ppo_v3_best_model.zip"),
        help="Path to trained PPO model (.zip or base path without extension).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "comparison"),
        help="Root directory for batched comparison outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional experiment tag for easier result identification.",
    )
    p.add_argument(
        "--strict-ppo",
        action="store_true",
        help="Fail immediately if PPO is requested but the model file does not exist.",
    )
    p.add_argument("--keepalive-steps", type=int, default=8, help="Container TTL (match training env)")
    p.add_argument("--beta-base", type=float, default=0.2, help="Dynamic beta base (match training)")
    p.add_argument("--beta-growth", type=float, default=3.0, help="Dynamic beta growth (match training)")
    p.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Reward scale (match training if you used --reward-scale in main_train.py)",
    )
    p.add_argument(
        "--vocab-days",
        type=str,
        default="",
        help="Comma-separated days used to build a fixed function/layer vocabulary. Default: try PPO model sidecar, else per-run dynamic vocab.",
    )
    p.add_argument(
        "--vocab-rows",
        type=int,
        default=0,
        help="Rows per day when building fixed vocabulary (0 = use full day).",
    )
    return p.parse_args()


def parse_csv_items(raw: str) -> List[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Argument must contain at least one non-empty item.")
    return items


def parse_seeds(raw: str) -> List[int]:
    try:
        return [int(item) for item in parse_csv_items(raw)]
    except ValueError as exc:
        raise ValueError(f"Invalid seed list: {raw}") from exc


def load_trace_for_day(day: str, rows: int) -> pd.DataFrame:
    parser = AzureTraceParser()
    trace_df = parser.load_day_data(day)
    if rows and rows > 0:
        trace_df = trace_df.iloc[:rows].copy()
    return trace_df


def make_run_root(output_root: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{tag}" if tag else timestamp
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def aggregate_across_runs(all_summary_df: pd.DataFrame, run_root: Path) -> None:
    """Save cross-run summary tables for later paper writing."""
    all_path = run_root / "all_runs_summary.csv"
    all_summary_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    group_cols = ["policy", "policy_key"]
    metric_cols = [
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "tail_gap_ms",
        "avg_memory_pressure",
        "peak_memory_pressure",
        "final_cache_hit_rate",
        "avg_reward",
        "invalid_ratio",
        "hot_ratio",
        "warm_ratio",
        "cold_ratio",
    ]
    grouped = (
        all_summary_df.groupby(group_cols, as_index=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([part for part in col if part]).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    grouped_path = run_root / "grouped_policy_summary.csv"
    grouped.to_csv(grouped_path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    days = parse_csv_items(args.days)
    seeds = parse_seeds(args.seeds)
    requested_policies = get_requested_policies(args.policies)
    model_path = resolve_model_path(args.model_path)
    vocab = None
    if args.vocab_days:
        vocab_days = parse_vocab_items(args.vocab_days)
        vocab = build_vocab_from_days(vocab_days, rows_per_day=args.vocab_rows)
    elif "ppo" in requested_policies and model_path is not None:
        vocab = load_vocab(vocab_sidecar_path(model_path))

    output_root = Path(args.output_root)
    run_root = make_run_root(output_root=output_root, tag=args.tag)

    print("[1/4] 解析实验配置...")
    print(f"    days={days}")
    print(f"    seeds={seeds}")
    print(f"    policies={requested_policies}")
    print(f"    run_root={run_root}")
    if vocab is not None:
        print(
            "    fixed_vocab="
            f"(functions={len(vocab['function_names'])}, layers={len(vocab['layer_names'])})"
        )

    print("[2/4] 加载策略定义...")
    loaded_policies = load_policies(requested_policies, model_path, args.strict_ppo)

    print("[3/4] 开始批量对比实验...")
    all_run_summaries = []

    for day in days:
        print(f"    -> 加载数据 {day} ...")
        trace_df = load_trace_for_day(day=day, rows=args.rows)
        print(f"       样本行数: {len(trace_df)}")

        for seed in seeds:
            run_name = f"{day}_seed{seed}"
            print(f"       -> 运行 {run_name} ...")

            metrics_dir = run_root / run_name / "metrics_csv"
            plots_dir = run_root / run_name / "plots"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            summaries = []
            step_frames = []
            env_common = {
                "keepalive_steps": args.keepalive_steps,
                "beta_base": args.beta_base,
                "beta_growth": args.beta_growth,
                "reward_scale": args.reward_scale,
            }
            if vocab is not None:
                env_common["function_names"] = vocab["function_names"]
                env_common["layer_names"] = vocab["layer_names"]

            for policy_key, policy_label, spec in loaded_policies:
                print(f"          -> 评估 {policy_label}")
                env = build_env(
                    trace_df=trace_df,
                    num_nodes=args.nodes,
                    seed=seed,
                    env_kwargs=env_common,
                )
                policy_obj = spec(env) if isinstance(spec, type) else spec

                if policy_key == "ppo":
                    validate_model_env_compatibility(
                        policy_label=policy_label,
                        policy_obj=policy_obj,
                        env=env,
                    )

                summary, step_df = evaluate_policy(
                    policy_key=policy_key,
                    policy_label=policy_label,
                    policy_obj=policy_obj,
                    env=env,
                )
                summary["day"] = day
                summary["seed"] = seed
                summary["run_name"] = run_name
                summaries.append(summary)
                step_frames.append(step_df)
                save_step_metrics(step_df, policy_key, metrics_dir)

            summary_df = pd.DataFrame(summaries).sort_values(
                by=["avg_latency_ms", "p95_latency_ms"], ascending=[True, True]
            )
            save_summary(summary_df, metrics_dir)
            save_text_report(summary_df, metrics_dir)
            plot_summary_bars(summary_df, plots_dir)
            plot_startup_mix(summary_df, plots_dir)
            plot_time_series(step_frames, plots_dir)

            all_run_summaries.append(summary_df)

    if not all_run_summaries:
        raise RuntimeError("No comparison runs were completed.")

    print("[4/4] 写入跨实验聚合汇总...")
    all_summary_df = pd.concat(all_run_summaries, ignore_index=True)
    aggregate_across_runs(all_summary_df=all_summary_df, run_root=run_root)

    print("\n批量对比实验完成。")
    print(f"- 运行目录: {run_root}")
    print(f"- 全部汇总: {run_root / 'all_runs_summary.csv'}")
    print(f"- 聚合汇总: {run_root / 'grouped_policy_summary.csv'}")


if __name__ == "__main__":
    main()
