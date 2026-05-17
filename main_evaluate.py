"""
入口 2：V3 统一评估脚本。

功能目标（对应《施工总览图 V3》）：
1. 加载训练好的 PPO 模型，并与 LRU / Greedy / Random 基线进行统一对比
2. 在同一份 Azure trace 与同一套环境参数下，输出逐步评估明细与汇总指标
3. 生成论文可直接使用的核心对比图表，服务于后续 run_comparison / run_ablation
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from algorithms import GreedyPolicy, LRUKeepalivePolicy, RandomPolicy
from envs.serverless_env import ServerlessEdgeEnv
from utils.metrics_tracker import (
    save_step_metrics,
    save_summary,
    save_text_report,
    summarize_step_records,
)
from utils.plotter import plot_startup_mix, plot_summary_bars, plot_time_series
from utils.trace_parser import AzureTraceParser
from utils.vocab import (
    build_vocab_from_days,
    build_vocab_from_frames,
    load_vocab,
    parse_csv_items,
    vocab_sidecar_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPO and heuristic baselines on ServerlessEdgeEnv (V3)")
    p.add_argument("--day", type=str, default="d01", help="Azure trace day, e.g. d01")
    p.add_argument(
        "--rows",
        type=int,
        default=20_000,
        help="Max rows from parsed trace (0 = use all). Smaller is faster for smoke tests.",
    )
    p.add_argument("--nodes", type=int, default=3, help="Number of edge nodes")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("results", "models", "ppo_v3_best_model.zip"),
        help="Path to trained PPO model (.zip or base path without extension).",
    )
    p.add_argument(
        "--policies",
        type=str,
        default="ppo,lru,greedy,random",
        help="Comma-separated policies to evaluate: ppo,lru,greedy,random",
    )
    p.add_argument(
        "--metrics-dir",
        type=str,
        default=os.path.join("results", "metrics_csv"),
        help="Directory to save evaluation CSV files.",
    )
    p.add_argument(
        "--plots-dir",
        type=str,
        default=os.path.join("results", "plots"),
        help="Directory to save evaluation figures.",
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
        help="Comma-separated days used to build a fixed function/layer vocabulary. Default: try model sidecar, else current evaluation slice.",
    )
    p.add_argument(
        "--vocab-rows",
        type=int,
        default=0,
        help="Rows per day when building fixed vocabulary (0 = use full day).",
    )
    return p.parse_args()


def resolve_model_path(model_path: str) -> Path | None:
    """Resolve PPO model path, supporting both with/without .zip suffix."""
    candidate = Path(model_path)
    if candidate.exists():
        return candidate

    if candidate.suffix != ".zip":
        zipped = candidate.with_suffix(".zip")
        if zipped.exists():
            return zipped

    if candidate.suffix == ".zip":
        no_suffix = candidate.with_suffix("")
        if no_suffix.exists():
            return no_suffix

    return None


def build_env(
    trace_df: pd.DataFrame,
    num_nodes: int,
    seed: int,
    env_kwargs: Dict | None = None,
) -> ServerlessEdgeEnv:
    """Construct a fresh evaluation environment."""
    return ServerlessEdgeEnv(
        trace_df=trace_df,
        num_nodes=num_nodes,
        seed=seed,
        **(env_kwargs or {}),
    )


def get_requested_policies(policies_arg: str) -> List[str]:
    allowed = {"ppo", "lru", "greedy", "random"}
    policies = [item.strip().lower() for item in policies_arg.split(",") if item.strip()]
    unknown = sorted(set(policies).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}. Allowed values: {sorted(allowed)}")
    if not policies:
        raise ValueError("No policies requested. Use --policies ppo,lru,greedy,random")
    return policies


def load_policies(requested: List[str], model_path: Path | None, strict_ppo: bool):
    """Load policy specifications. PPO may be skipped if its model is absent."""
    loaded: List[Tuple[str, str, object]] = []

    if "ppo" in requested:
        if model_path is None:
            message = (
                "未找到 PPO 模型文件，已跳过 PPO 评估。"
                "如需启用，请先运行 main_train.py 生成 results/models/ppo_v3_best_model.zip"
            )
            if strict_ppo:
                raise FileNotFoundError(message)
            print(f"[warn] {message}")
        else:
            import torch
            from stable_baselines3 import PPO

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[load] 正在加载 PPO 模型: {model_path} (device={device})")
            loaded.append(("ppo", "PPO", PPO.load(str(model_path), device=device)))

    if "lru" in requested:
        loaded.append(("lru", "LRU", LRUKeepalivePolicy))
    if "greedy" in requested:
        loaded.append(("greedy", "Greedy", GreedyPolicy))
    if "random" in requested:
        loaded.append(("random", "Random", RandomPolicy))

    if not loaded:
        raise ValueError("没有可评估的策略。请检查 --policies 参数或 PPO 模型路径。")
    return loaded


def _space_signature(space) -> object:
    """Build a lightweight signature for comparing gym spaces."""
    if hasattr(space, "spaces"):
        return (
            space.__class__.__name__,
            {key: _space_signature(subspace) for key, subspace in space.spaces.items()},
        )
    return (
        space.__class__.__name__,
        tuple(getattr(space, "shape", ())),
        str(getattr(space, "dtype", "")),
    )


def validate_model_env_compatibility(policy_label: str, policy_obj, env: ServerlessEdgeEnv) -> None:
    """
    Validate that a loaded PPO model matches the current environment definition.

    This is especially important when the environment observation space changed
    after the model was trained, which would otherwise fail deep inside SB3.
    """
    if not hasattr(policy_obj, "observation_space") or not hasattr(policy_obj, "action_space"):
        return

    model_obs_sig = _space_signature(policy_obj.observation_space)
    env_obs_sig = _space_signature(env.observation_space)
    model_action_sig = _space_signature(policy_obj.action_space)
    env_action_sig = _space_signature(env.action_space)

    if model_obs_sig != env_obs_sig or model_action_sig != env_action_sig:
        raise ValueError(
            f"{policy_label} 模型与当前环境不兼容。\n"
            "这通常说明你现在加载的是旧版本模型，而当前 `ServerlessEdgeEnv` 已经改成了新的观测空间。\n"
            f"- 模型 observation_space: {policy_obj.observation_space}\n"
            f"- 当前环境 observation_space: {env.observation_space}\n"
            f"- 模型 action_space: {policy_obj.action_space}\n"
            f"- 当前环境 action_space: {env.action_space}\n\n"
            "解决方法：请用当前代码重新运行 `main_train.py` 训练一个新 PPO 模型，"
            "并确保训练与评估使用同一份固定词表（例如传入相同的 `--vocab-days`，"
            "或复用训练时保存的 `_vocab.json` sidecar）。"
        )


def evaluate_policy(policy_key: str, policy_label: str, policy_obj, env: ServerlessEdgeEnv):
    """
    Run one full evaluation episode on a fresh environment.

    Returns
    -------
    summary : dict
        汇总指标
    step_df : pd.DataFrame
        每一步的详细评估记录
    """
    # Gymnasium 要求 seed 为 Python int，numpy.int64 会触发 Error
    obs, _ = env.reset(seed=int(env.rng.integers(0, 1_000_000)))
    terminated = False
    truncated = False
    step_idx = 0
    total_reward = 0.0
    step_records: List[Dict[str, float | int | str]] = []

    while not terminated and not truncated:
        action, _ = policy_obj.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).item())

        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += float(reward)

        step_records.append(
            {
                "policy": policy_label,
                "policy_key": policy_key,
                "step": step_idx,
                "action_node": action_int,
                "reward": float(reward),
                "latency_ms": float(info["latency_ms"]),
                "avg_latency_ms": float(info["avg_latency_ms"]),
                "memory_pressure": float(info["memory_pressure"]),
                "cache_hit_rate": float(info["cache_hit_rate"]),
                "start_state": str(info["start_state"]),
                "invalid_action": int(info["start_state"] == "invalid"),
                "hot_hits_cum": float(info["hot_hits"]),
                "warm_hits_cum": float(info["warm_hits"]),
                "cold_hits_cum": float(info["cold_hits"]),
            }
        )
        step_idx += 1

    step_df = pd.DataFrame(step_records)
    summary = summarize_step_records(
        step_df=step_df,
        policy_key=policy_key,
        policy_label=policy_label,
        total_reward=total_reward,
    )
    return summary, step_df


def main():
    args = parse_args()
    requested_policies = get_requested_policies(args.policies)

    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 初始化数据管道...")
    parser = AzureTraceParser()
    trace_df = parser.load_day_data(args.day)
    if args.rows and args.rows > 0:
        trace_df = trace_df.iloc[: args.rows].copy()
    print(f"    评估样本行数: {len(trace_df)}")

    print("[2/5] 解析待评估策略...")
    model_path = resolve_model_path(args.model_path)
    loaded_policies = load_policies(requested_policies, model_path, args.strict_ppo)
    print(f"    已启用策略: {[label for _, label, _ in loaded_policies]}")

    vocab = None
    if args.vocab_days:
        vocab_days = parse_csv_items(args.vocab_days)
        print(f"    固定词表来源 days={vocab_days}, vocab_rows={args.vocab_rows}")
        vocab = build_vocab_from_days(vocab_days, rows_per_day=args.vocab_rows)
    elif "ppo" in requested_policies and model_path is not None:
        sidecar_path = vocab_sidecar_path(model_path)
        vocab = load_vocab(sidecar_path)
        if vocab is not None:
            print(f"    已加载模型固定词表: {sidecar_path}")

    if vocab is None:
        vocab = build_vocab_from_frames([trace_df])
        print("    未指定固定词表，使用当前评估切片构建词表。")

    print(
        "    固定词表大小: "
        f"functions={len(vocab['function_names'])}, layers={len(vocab['layer_names'])}"
    )

    print("[3/5] 逐策略执行完整评估...")
    summaries = []
    step_frames = []
    saved_step_paths = []

    env_common = {
        "keepalive_steps": args.keepalive_steps,
        "beta_base": args.beta_base,
        "beta_growth": args.beta_growth,
        "reward_scale": args.reward_scale,
        "function_names": vocab["function_names"],
        "layer_names": vocab["layer_names"],
    }
    for policy_key, policy_label, spec in loaded_policies:
        print(f"    -> 正在评估 {policy_label} ...")
        env = build_env(
            trace_df=trace_df,
            num_nodes=args.nodes,
            seed=args.seed,
            env_kwargs=env_common,
        )
        policy_obj = spec(env) if isinstance(spec, type) else spec
        if policy_key == "ppo":
            validate_model_env_compatibility(policy_label=policy_label, policy_obj=policy_obj, env=env)

        summary, step_df = evaluate_policy(
            policy_key=policy_key,
            policy_label=policy_label,
            policy_obj=policy_obj,
            env=env,
        )
        summaries.append(summary)
        step_frames.append(step_df)
        saved_step_paths.append(save_step_metrics(step_df, policy_key, metrics_dir))

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["avg_latency_ms", "p95_latency_ms"], ascending=[True, True]
    )

    print("[4/5] 导出汇总结果与报告...")
    summary_path = save_summary(summary_df, metrics_dir)
    report_path = save_text_report(summary_df, metrics_dir)

    print("[5/5] 生成对比图表...")
    summary_plot = plot_summary_bars(summary_df, plots_dir)
    startup_plot = plot_startup_mix(summary_df, plots_dir)
    latency_plot, memory_plot = plot_time_series(step_frames, plots_dir)

    print("\n评估完成。核心输出如下：")
    print(f"- 汇总表: {summary_path}")
    print(f"- 文本报告: {report_path}")
    for step_path in saved_step_paths:
        print(f"- 逐步明细: {step_path}")
    print(f"- 图表 1 (核心指标): {summary_plot}")
    print(f"- 图表 2 (启动结构): {startup_plot}")
    print(f"- 图表 3 (时延轨迹): {latency_plot}")
    print(f"- 图表 4 (内存压力轨迹): {memory_plot}")

    print("\n策略表现概览：")
    display_cols = [
        "policy",
        "avg_latency_ms",
        "p95_latency_ms",
        "final_cache_hit_rate",
        "avg_memory_pressure",
        "invalid_requests",
    ]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
