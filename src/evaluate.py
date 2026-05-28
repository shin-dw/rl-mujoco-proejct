"""
평가 및 시각화 스크립트

저장된 모델을 로드하여 평가하고 결과를 시각화합니다.

사용 예시:
    python -m src.evaluate --model results/sac_HalfCheetah-v5_seed42/models/model_final.pt --algo sac --env HalfCheetah-v5
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

from .algorithms.ppo import PPO
from .algorithms.sac import SAC
from .algorithms.td3 import TD3
from .common.env_wrapper import NormalizeObservation


def parse_args():
    parser = argparse.ArgumentParser(description="모델 평가 및 시각화")
    parser.add_argument("--model", type=str, required=True, help="모델 파일 경로")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=20, help="평가 에피소드 수")
    parser.add_argument("--render", action="store_true", help="렌더링 활성화")
    parser.add_argument("--save-video", action="store_true", help="영상 저장")
    parser.add_argument("--output-dir", type=str, default="results/eval")
    return parser.parse_args()


def create_algo_for_eval(algo_name, obs_dim, act_dim):
    """평가용 알고리즘 인스턴스 생성."""
    if algo_name == "ppo":
        return PPO(obs_dim, act_dim)
    elif algo_name == "sac":
        return SAC(obs_dim, act_dim)
    elif algo_name == "td3":
        return TD3(obs_dim, act_dim)
    raise ValueError(f"지원하지 않는 알고리즘: {algo_name}")


def load_obs_stats(model_path: str):
    """모델과 함께 저장된 obs 정규화 통계를 불러옵니다."""
    stats_path = model_path.replace(".pt", "_obs_stats.npz")
    if not os.path.exists(stats_path):
        return None
    data = np.load(stats_path)
    return {
        "running_mean": data["running_mean"],
        "running_var": data["running_var"],
        "count": int(data["count"][0]),
    }


def make_eval_env(env_id: str, model_path: str, render_mode=None) -> gym.Env:
    """obs 통계가 있으면 NormalizeObservation을 복원한 평가 환경을 생성합니다."""
    env = gym.make(env_id, render_mode=render_mode)
    obs_stats = load_obs_stats(model_path)
    if obs_stats is not None:
        env = NormalizeObservation(env)
        env.running_mean = obs_stats["running_mean"]
        env.running_var = obs_stats["running_var"]
        env.count = obs_stats["count"]
        print(f"  [정규화] obs 통계 복원 완료 (count={obs_stats['count']:,})")
    else:
        print("  [경고] obs 통계 파일(_obs_stats.npz)을 찾을 수 없습니다. raw obs로 평가합니다.")
    return env


def evaluate_model(algo, env, n_episodes: int) -> dict:
    """모델을 평가하고 결과를 반환합니다."""
    returns, lengths = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return, ep_length = 0.0, 0

        while not done:
            action = algo.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_length)
        print(f"  에피소드 {ep+1}/{n_episodes}: Return={ep_return:.1f}, Length={ep_length}")

    return {
        "returns": returns,
        "lengths": lengths,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_length": np.mean(lengths),
    }


def plot_training_curve(log_dir: str, output_path: str):
    """학습 곡선을 그립니다."""
    csv_path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(csv_path):
        print(f"  CSV 파일을 찾을 수 없습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 에피소드 리턴
    if "episode/return" in df.columns and "step" in df.columns:
        axes[0].plot(df["step"], df["episode/return"], alpha=0.7)
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Episode Return")
        axes[0].set_title("학습 곡선 (Episode Return)")
        axes[0].grid(True, alpha=0.3)

    # 평가 리턴
    if "eval/mean_return" in df.columns and "step" in df.columns:
        eval_df = df.dropna(subset=["eval/mean_return"])
        axes[1].plot(eval_df["step"], eval_df["eval/mean_return"], "o-")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Eval Mean Return")
        axes[1].set_title("평가 곡선 (Eval Return)")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  학습 곡선 저장: {output_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 환경 생성 (obs 통계 복원 포함)
    render_mode = "human" if args.render else None
    env = make_eval_env(args.env, args.model, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 알고리즘 생성 및 모델 로드
    algo = create_algo_for_eval(args.algo, obs_dim, act_dim)
    algo.load(args.model)

    # 평가
    print(f"\n{'='*40}")
    print(f"  {args.algo.upper()} on {args.env}")
    print(f"{'='*40}")

    results = evaluate_model(algo, env, args.episodes)

    print(f"\n{'='*40}")
    print(f"  평균 Return: {results['mean_return']:.1f} ± {results['std_return']:.1f}")
    print(f"  평균 Length: {results['mean_length']:.0f}")
    print(f"{'='*40}")

    # 평가 결과를 CSV로 저장
    import csv
    csv_path = os.path.join(args.output_dir, f"{args.algo}_{args.env}_eval.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length"])
        for i, (ret, length) in enumerate(zip(results["returns"], results["lengths"])):
            writer.writerow([i + 1, ret, length])
    print(f"\n  평가 결과 저장: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
