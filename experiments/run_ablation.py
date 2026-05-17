"""
自动化消融实验入口。

对应《施工总览图 V3》的实验设计：
1. 完整 V3 算法
2. 去掉镜像层共享（退回无温启动的简单冷启动）
3. 将动态 beta 改为固定 beta

本脚本会：
- 分别训练三种 PPO 变体
- 在指定评估集上统一评估三种变体
- 导出逐步明细、汇总表、文字报告与对比图
- 生成跨 day / seed 的聚合汇总结果
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_evaluate import (  # noqa: E402
    build_env,
    evaluate_policy,
    validate_model_env_compatibility,
)
from utils.metrics_tracker import save_step_metrics, save_summary, save_text_report  # noqa: E402
from utils.plotter import plot_startup_mix, plot_summary_bars, plot_time_series  # noqa: E402
from utils.trace_parser import load_concat_trace_days  # noqa: E402
from utils.vocab import build_vocab_from_days, save_vocab, vocab_sidecar_path  # noqa: E402


VARIANT_LABELS = {
    "full_v3": "FullV3",
    "no_layer_sharing": "NoLayerSharing",
    "fixed_beta": "FixedBeta",
}


def parse_args():
    p = argparse.ArgumentParser(description="Run PPO ablation experiments for V3 environment.")
    p.add_argument(
        "--train-days",
        type=str,
        default="d01",
        help="Comma-separated training days, e.g. d01,d02,d03",
    )
    p.add_argument(
        "--eval-days",
        type=str,
        default="d01",
        help="Comma-separated evaluation days, e.g. d13,d14",
    )
    p.add_argument(
        "--train-rows",
        type=int,
        default=20_000,
        help="Max rows loaded per training day (0 = use all).",
    )
    p.add_argument(
        "--eval-rows",
        type=int,
        default=20_000,
        help="Max rows loaded per evaluation day (0 = use all).",
    )
    p.add_argument("--nodes", type=int, default=3, help="Number of edge nodes")
    p.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds, e.g. 42,43,44",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total PPO training timesteps for each ablation variant.",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency during training.",
    )
    p.add_argument(
        "--fixed-beta",
        type=float,
        default=0.2,
        help="Fixed beta value used by the fixed-beta ablation.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "ablation"),
        help="Root directory for ablation outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional experiment tag for easier identification.",
    )
    p.add_argument(
        "--reuse-models",
        action="store_true",
        help="Reuse previously saved ablation models if they already exist.",
    )
    p.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    p.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps per rollout")
    p.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    p.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor")
    p.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy coefficient")
    p.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
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
        help="Comma-separated days used to build a fixed function/layer vocabulary. Default: union of train-days and eval-days.",
    )
    p.add_argument(
        "--vocab-rows",
        type=int,
        default=-1,
        help="Rows per day when building fixed vocabulary (-1 = auto from train/eval rows, 0 = use full day).",
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


def load_trace_for_days(days: List[str], rows_per_day: int) -> pd.DataFrame:
    return load_concat_trace_days(days=days, rows_per_day=rows_per_day)


def make_run_root(output_root: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{tag}" if tag else timestamp
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def get_variant_specs(fixed_beta: float) -> List[Dict]:
    return [
        {
            "variant_key": "full_v3",
            "variant_label": VARIANT_LABELS["full_v3"],
            "env_kwargs": {
                "enable_layer_sharing": True,
                "enable_dynamic_beta": True,
            },
        },
        {
            "variant_key": "no_layer_sharing",
            "variant_label": VARIANT_LABELS["no_layer_sharing"],
            "env_kwargs": {
                "enable_layer_sharing": False,
                "enable_dynamic_beta": True,
            },
        },
        {
            "variant_key": "fixed_beta",
            "variant_label": VARIANT_LABELS["fixed_beta"],
            "env_kwargs": {
                "enable_layer_sharing": True,
                "enable_dynamic_beta": False,
                "fixed_beta_value": fixed_beta,
            },
        },
    ]


def train_variant_model(
    train_df: pd.DataFrame,
    nodes: int,
    seed: int,
    timesteps: int,
    save_freq: int,
    variant_key: str,
    variant_label: str,
    env_kwargs: Dict,
    variant_dir: Path,
    reuse_models: bool,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    gamma: float,
    ent_coef: float,
    clip_range: float,
    vocab: Dict[str, List[str]],
) -> Path:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    models_dir = variant_dir / "models"
    checkpoints_dir = models_dir / "checkpoints"
    monitor_dir = variant_dir / "tensorboard_logs" / "monitor"
    tensorboard_dir = variant_dir / "tensorboard_logs" / variant_key

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = models_dir / f"{variant_key}_ppo_model.zip"
    if reuse_models and final_model_path.exists():
        print(f"       [reuse] 复用已有模型: {final_model_path}")
        sidecar_path = vocab_sidecar_path(final_model_path)
        if not sidecar_path.exists():
            save_vocab(vocab=vocab, target_path=sidecar_path)
        return final_model_path

    def make_env():
        env = build_env(trace_df=train_df, num_nodes=nodes, seed=seed, env_kwargs=env_kwargs)
        return Monitor(env, filename=str(monitor_dir / "train"))

    vec_env = DummyVecEnv([make_env])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"       [train] {variant_label} 开始训练 "
        f"(timesteps={timesteps}, device={device}, seed={seed})"
    )
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
        device=device,
        seed=seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq, 1),
        save_path=str(checkpoints_dir),
        name_prefix=variant_key,
    )
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=True)

    model.save(str(final_model_path.with_suffix("")))
    save_vocab(vocab=vocab, target_path=vocab_sidecar_path(final_model_path))
    print(f"       [train] {variant_label} 训练完成: {final_model_path}")
    return final_model_path


def evaluate_variant_model(
    model_path: Path,
    eval_df: pd.DataFrame,
    nodes: int,
    seed: int,
    variant_key: str,
    variant_label: str,
    env_kwargs: Dict,
):
    from stable_baselines3 import PPO

    env = build_env(trace_df=eval_df, num_nodes=nodes, seed=seed, env_kwargs=env_kwargs)
    model = PPO.load(str(model_path), device="cpu")
    validate_model_env_compatibility(policy_label=variant_label, policy_obj=model, env=env)
    return evaluate_policy(
        policy_key=variant_key,
        policy_label=variant_label,
        policy_obj=model,
        env=env,
    )


def aggregate_across_runs(all_summary_df: pd.DataFrame, run_root: Path) -> None:
    all_path = run_root / "all_ablation_summary.csv"
    all_summary_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    group_cols = ["policy", "policy_key", "eval_day"]
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
    grouped.to_csv(run_root / "grouped_ablation_summary.csv", index=False, encoding="utf-8-sig")


def save_variant_manifest(run_root: Path, variant_specs: List[Dict], train_days: List[str], eval_days: List[str]) -> None:
    manifest_rows = []
    for spec in variant_specs:
        manifest_rows.append(
            {
                "variant_key": spec["variant_key"],
                "variant_label": spec["variant_label"],
                "env_kwargs": str(spec["env_kwargs"]),
                "train_days": ",".join(train_days),
                "eval_days": ",".join(eval_days),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        run_root / "ablation_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    args = parse_args()
    train_days = parse_csv_items(args.train_days)
    eval_days = parse_csv_items(args.eval_days)
    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root)
    run_root = make_run_root(output_root=output_root, tag=args.tag)
    variant_specs = get_variant_specs(fixed_beta=args.fixed_beta)
    vocab_days = parse_csv_items(args.vocab_days) if args.vocab_days else list(dict.fromkeys(train_days + eval_days))
    if args.vocab_rows >= 0:
        vocab_rows = args.vocab_rows
    elif args.train_rows == 0 or args.eval_rows == 0:
        vocab_rows = 0
    else:
        vocab_rows = max(args.train_rows, args.eval_rows)
    vocab = build_vocab_from_days(vocab_days, rows_per_day=vocab_rows)

    print("[1/5] 解析实验配置...")
    print(f"    train_days={train_days}")
    print(f"    eval_days={eval_days}")
    print(f"    seeds={seeds}")
    print(f"    run_root={run_root}")
    print(
        "    fixed_vocab="
        f"(days={vocab_days}, rows={vocab_rows}, functions={len(vocab['function_names'])}, layers={len(vocab['layer_names'])})"
    )
    save_variant_manifest(run_root, variant_specs, train_days, eval_days)

    print("[2/5] 加载训练数据...")
    train_df = load_trace_for_days(days=train_days, rows_per_day=args.train_rows)
    print(f"    训练数据总行数: {len(train_df)}")

    print("[3/5] 训练三组消融变体...")
    trained_models: Dict[tuple, tuple] = {}
    env_base_kwargs = {
        "keepalive_steps": args.keepalive_steps,
        "beta_base": args.beta_base,
        "beta_growth": args.beta_growth,
        "reward_scale": args.reward_scale,
        "function_names": vocab["function_names"],
        "layer_names": vocab["layer_names"],
    }
    for seed in seeds:
        for spec in variant_specs:
            variant_key = spec["variant_key"]
            variant_label = spec["variant_label"]
            env_kwargs = {**env_base_kwargs, **spec["env_kwargs"]}
            variant_dir = run_root / "training" / f"{variant_key}_seed{seed}"
            model_path = train_variant_model(
                train_df=train_df,
                nodes=args.nodes,
                seed=seed,
                timesteps=args.timesteps,
                save_freq=args.save_freq,
                variant_key=variant_key,
                variant_label=variant_label,
                env_kwargs=env_kwargs,
                variant_dir=variant_dir,
                reuse_models=args.reuse_models,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                clip_range=args.clip_range,
                vocab=vocab,
            )
            trained_models[(variant_key, seed)] = (model_path, env_kwargs, variant_label)

    print("[4/5] 在评估集上运行消融对比...")
    all_run_summaries = []
    for eval_day in eval_days:
        eval_df = load_trace_for_days(days=[eval_day], rows_per_day=args.eval_rows)
        print(f"    -> 评估数据 {eval_day}, 行数={len(eval_df)}")

        for seed in seeds:
            run_name = f"{eval_day}_seed{seed}"
            metrics_dir = run_root / "evaluations" / run_name / "metrics_csv"
            plots_dir = run_root / "evaluations" / run_name / "plots"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            summaries = []
            step_frames = []
            for spec in variant_specs:
                variant_key = spec["variant_key"]
                model_path, env_kwargs, variant_label = trained_models[(variant_key, seed)]
                print(f"       -> {run_name} 评估 {variant_label}")
                summary, step_df = evaluate_variant_model(
                    model_path=model_path,
                    eval_df=eval_df,
                    nodes=args.nodes,
                    seed=seed,
                    variant_key=variant_key,
                    variant_label=variant_label,
                    env_kwargs=env_kwargs,
                )
                summary["train_days"] = ",".join(train_days)
                summary["eval_day"] = eval_day
                summary["seed"] = seed
                summary["run_name"] = run_name
                summaries.append(summary)
                step_frames.append(step_df)
                save_step_metrics(step_df, variant_key, metrics_dir)

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
        raise RuntimeError("No ablation runs were completed.")

    print("[5/5] 写入跨实验聚合汇总...")
    all_summary_df = pd.concat(all_run_summaries, ignore_index=True)
    aggregate_across_runs(all_summary_df=all_summary_df, run_root=run_root)

    print("\n消融实验完成。")
    print(f"- 运行目录: {run_root}")
    print(f"- 全部汇总: {run_root / 'all_ablation_summary.csv'}")
    print(f"- 聚合汇总: {run_root / 'grouped_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
