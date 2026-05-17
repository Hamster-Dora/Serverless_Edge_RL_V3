"""
入口 1：V3 PPO 训练。观察空间为 Dict，必须使用 MultiInputPolicy。
"""
import argparse
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.serverless_env import ServerlessEdgeEnv
from utils.trace_parser import AzureTraceParser, load_concat_trace_days
from utils.vocab import build_vocab_from_days, build_vocab_from_frames, parse_csv_items, save_vocab, vocab_sidecar_path


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on ServerlessEdgeEnv (V3)")
    p.add_argument("--day", type=str, default="d01", help="Azure trace day (single-day mode), e.g. d01")
    p.add_argument(
        "--train-days",
        type=str,
        default="",
        help="Comma-separated training days for multi-day PPO training; non-empty overrides --day "
        "(data are concatenated in day order).",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=50_000,
        help="Max rows from parsed trace (0 = use all). Smaller = faster smoke test.",
    )
    p.add_argument("--nodes", type=int, default=3, help="Number of edge nodes")
    p.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total PPO training timesteps",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default=os.path.join("results", "models", "ppo_v3_best_model"),
        help=(
            'Base path for Stable-Baselines3 model.save (omit ".zip"; SB3 adds it automatically). '
            'Phase 6 nightly / final runs commonly use results/models/ppo_phase6_final.'
        ),
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency (steps)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    # PPO 超参（阶段 2 调参用，无需改代码）
    p.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    p.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps per rollout")
    p.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    p.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor")
    p.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy coefficient (exploration)")
    p.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    # 环境参数（与数学建模中动态 β、保活、奖励尺度一致）
    p.add_argument("--keepalive-steps", type=int, default=8, help="Container TTL in steps")
    p.add_argument("--beta-base", type=float, default=0.2, help="Dynamic beta base")
    p.add_argument("--beta-growth", type=float, default=3.0, help="Dynamic beta growth with memory pressure")
    p.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiply all step rewards (e.g. 0.001) to stabilize critic when magnitudes are huge",
    )
    p.add_argument(
        "--vocab-days",
        type=str,
        default="",
        help="Comma-separated days used to build a fixed function/layer vocabulary. Default: current training slice only.",
    )
    p.add_argument(
        "--vocab-rows",
        type=int,
        default=0,
        help="Rows per day when building fixed vocabulary (0 = use full day).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("[1/4] 初始化数据管道...")
    parser_tool = AzureTraceParser()

    if args.train_days.strip():
        train_day_list = parse_csv_items(args.train_days)
        df_train = load_concat_trace_days(days=train_day_list, rows_per_day=args.rows)
        train_day_meta = ",".join(train_day_list)
    else:
        train_day_list = [args.day]
        df_train = parser_tool.load_day_data(args.day)
        if args.rows and args.rows > 0:
            df_train = df_train.iloc[: args.rows].copy()
        train_day_meta = args.day

    print(f"    训练样本行数: {len(df_train)}")

    if args.vocab_days:
        vocab_days = parse_csv_items(args.vocab_days)
        effective_vocab_rows = args.vocab_rows
        if (
            args.rows
            and args.rows > 0
            and args.vocab_rows > 0
            and set(train_day_list).issubset(set(vocab_days))
            and args.vocab_rows < args.rows
        ):
            effective_vocab_rows = args.rows
            print(
                "    [warn] `vocab_rows` 小于训练切片 `rows`，"
                f"自动提升为 {effective_vocab_rows} 以覆盖训练数据中的全部函数/层。"
            )
        print(f"    固定词表来源 days={vocab_days}, vocab_rows={effective_vocab_rows}")
        vocab = build_vocab_from_days(vocab_days, rows_per_day=effective_vocab_rows)
    else:
        vocab = build_vocab_from_frames([df_train])

    print(
        "    固定词表大小: "
        f"functions={len(vocab['function_names'])}, layers={len(vocab['layer_names'])}"
    )

    print(f"[2/4] 构建 V3 环境 (num_nodes={args.nodes})...")

    def make_env():
        env = ServerlessEdgeEnv(
            trace_df=df_train,
            function_names=vocab["function_names"],
            layer_names=vocab["layer_names"],
            num_nodes=args.nodes,
            seed=args.seed,
            keepalive_steps=args.keepalive_steps,
            beta_base=args.beta_base,
            beta_growth=args.beta_growth,
            reward_scale=args.reward_scale,
        )
        # Monitor 写入 episode 统计，便于 TensorBoard 看回报曲线
        log_dir = os.path.join("results", "tensorboard_logs", "monitor")
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, filename=os.path.join(log_dir, "train"))

    vec_env = DummyVecEnv([make_env])

    os.makedirs(os.path.join("results", "models", "checkpoints"), exist_ok=True)
    tb_root = os.path.join("results", "tensorboard_logs", "ppo_v3_tensorboard")
    os.makedirs(tb_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/4] 初始化 PPO (device={device}, MultiInputPolicy)...")

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        tensorboard_log=tb_root,
        verbose=1,
        device=device,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=os.path.join("results", "models", "checkpoints"),
        name_prefix="ppo_v3",
    )

    print(f"[4/4] 开始训练 total_timesteps={args.timesteps} ...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    out_base = Path(args.save_path)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_path = os.fspath(out_base.with_suffix("")) if str(out_base.suffix).lower() == ".zip" else os.fspath(out_base)

    model.save(out_path)
    vocab_path = vocab_sidecar_path(out_path)
    save_vocab(
        vocab=vocab,
        target_path=vocab_path,
        metadata={
            "train_day": train_day_meta,
            "train_days": train_day_list,
            "train_rows_cap": args.rows,
            "vocab_days": parse_csv_items(args.vocab_days) if args.vocab_days else train_day_list,
            "vocab_rows": effective_vocab_rows if args.vocab_days else args.vocab_rows,
        },
    )
    print(f"训练完成。模型已保存: {out_path}.zip")
    print(f"固定词表已保存: {vocab_path}")
    print(f"TensorBoard: tensorboard --logdir {tb_root}")


if __name__ == "__main__":
    main()
