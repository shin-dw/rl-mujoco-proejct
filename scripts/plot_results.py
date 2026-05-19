"""
시각화 자동화 스크립트 — 팀원 A (환경/실험 인프라 담당) 제작

학습 로그(progress.csv)와 평가 결과(summary.csv)를 읽어
보고서용 그래프를 자동으로 생성합니다.

사용법:
    python scripts/plot_results.py                         # 전체 시각화
    python scripts/plot_results.py --type curves           # 학습 곡선만
    python scripts/plot_results.py --type bar              # 최종 성능 바 차트만
    python scripts/plot_results.py --type reward           # Reward Shaping 비교
    python scripts/plot_results.py --type dr               # Domain Randomization 비교
    python scripts/plot_results.py --env HalfCheetah-v5    # 특정 환경만
"""

import argparse, os, sys, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

ALGO_COLORS = {"PPO": "#1f77b4", "SAC": "#ff7f0e", "TD3": "#2ca02c"}
ALGO_ORDER = ["PPO", "SAC", "TD3"]
ENVS = ["HalfCheetah-v5", "Ant-v5", "Humanoid-v5"]

EXP_PATTERN = re.compile(
    r"^(?P<algo>ppo|sac|td3)_(?P<env>[A-Za-z]+-v\d+)"
    r"(?:_(?P<reward>[a-z_]+?))?(?P<dr>_dr)?_seed(?P<seed>\d+)$"
)


def discover_logs(results_dir, env_filter=None):
    """results/ 하위의 모든 progress.csv를 탐색하고 메타데이터와 함께 반환."""
    logs = []
    for exp_dir in sorted(Path(results_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        match = EXP_PATTERN.match(exp_dir.name)
        if not match:
            continue
        csv_path = exp_dir / "logs" / "progress.csv"
        if not csv_path.exists() or csv_path.stat().st_size < 200:
            continue
        info = match.groupdict()
        info["seed"] = int(info["seed"])
        info["dr"] = info["dr"] is not None
        if env_filter and info["env"] != env_filter:
            continue
        info["csv_path"] = str(csv_path)
        info["name"] = exp_dir.name
        logs.append(info)
    return logs


def smooth(data, window=10):
    """이동 평균으로 노이즈를 줄입니다."""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window, min_periods=1).mean().values


# ─── 1. 학습 곡선 비교 ───

def plot_training_curves(results_dir, output_dir, env_filter=None):
    """알고리즘별 학습 곡선을 환경별로 비교합니다 (다중 시드 평균 ± 표준편차)."""
    logs = discover_logs(results_dir, env_filter)
    if not logs:
        print("  [학습 곡선] 로그 데이터가 없습니다.")
        return

    # 환경별로 그룹핑
    env_groups = defaultdict(lambda: defaultdict(list))
    for log in logs:
        if log["reward"] or log["dr"]:
            continue  # 기본 실험만
        env_groups[log["env"]][log["algo"].upper()].append(log)

    for env_name, algo_dict in env_groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo_name in ALGO_ORDER:
            if algo_name not in algo_dict:
                continue
            all_returns = []
            for log in algo_dict[algo_name]:
                df = pd.read_csv(log["csv_path"])
                if "episode/return" in df.columns:
                    all_returns.append(df["episode/return"].dropna().values)

            if not all_returns:
                continue

            # 길이 맞추기 (가장 짧은 것 기준)
            min_len = min(len(r) for r in all_returns)
            trimmed = [r[:min_len] for r in all_returns]
            mean = smooth(np.mean(trimmed, axis=0), window=20)
            std = smooth(np.std(trimmed, axis=0), window=20)
            steps = np.arange(len(mean))

            color = ALGO_COLORS.get(algo_name, "#999999")
            ax.plot(steps, mean, color=color, label=algo_name, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Log Step")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"{env_name} — Algorithm Comparison")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, f"curve_{env_name}.png")
        plt.savefig(path)
        plt.close()
        print(f"  ✅ 학습 곡선: {path}")


# ─── 2. 최종 성능 바 차트 ───

def plot_performance_bar(results_dir, output_dir):
    """summary.csv를 읽어 알고리즘별 최종 성능 바 차트를 생성합니다."""
    summary_path = os.path.join(results_dir, "eval", "summary.csv")
    if not os.path.exists(summary_path):
        print(f"  [바 차트] summary.csv가 없습니다. 먼저 eval_all.py를 실행하세요.")
        return

    df = pd.read_csv(summary_path)
    df["mean_return"] = pd.to_numeric(df["mean_return"])
    df["std_return"] = pd.to_numeric(df["std_return"])

    # 기본 실험만 (default reward, no DR)
    base = df[(df["reward_type"] == "default") & (df["domain_rand"] == False)]
    if base.empty:
        print("  [바 차트] 기본 실험 데이터가 없습니다.")
        return

    envs = base["environment"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        env_data = base[base["environment"] == env]
        # 알고리즘별 평균
        grouped = env_data.groupby("algorithm").agg(
            mean=("mean_return", "mean"), std=("mean_return", "std")
        ).reindex([a for a in ALGO_ORDER if a in env_data["algorithm"].values])

        colors = [ALGO_COLORS.get(a, "#999") for a in grouped.index]
        bars = ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"],
                      color=colors, capsize=5, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax.set_title(env)
        ax.set_ylabel("Mean Return")
        ax.grid(axis="y", alpha=0.3)

        # 값 라벨
        for bar, val in zip(bars, grouped["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Algorithm Performance Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "performance_bar.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✅ 성능 바 차트: {path}")


# ─── 3. Reward Shaping 비교 ───

def plot_reward_comparison(results_dir, output_dir, env_filter=None):
    """기본 vs 커스텀 리워드 학습 곡선 비교."""
    logs = discover_logs(results_dir, env_filter)
    if not logs:
        print("  [Reward Shaping] 로그 데이터가 없습니다.")
        return

    # 리워드 타입별 그룹핑
    groups = defaultdict(list)
    for log in logs:
        if log["dr"]:
            continue
        key = (log["env"], log["algo"].upper())
        label = log["reward"] or "default"
        groups[key].append((label, log))

    if not any(len(v) > 1 for v in groups.values()):
        print("  [Reward Shaping] 비교할 데이터가 부족합니다 (기본 + 커스텀 모두 필요).")
        return

    for (env, algo), entries in groups.items():
        if len(entries) < 2:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, log in entries:
            df = pd.read_csv(log["csv_path"])
            if "episode/return" not in df.columns:
                continue
            returns = smooth(df["episode/return"].dropna().values, window=20)
            style = "-" if label == "default" else "--"
            ax.plot(returns, label=f"{label}", linestyle=style, linewidth=2)

        ax.set_xlabel("Log Step")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"{env} / {algo} — Reward Shaping Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, f"reward_{env}_{algo}.png")
        plt.savefig(path)
        plt.close()
        print(f"  ✅ Reward Shaping: {path}")


# ─── 4. Domain Randomization 비교 ───

def plot_dr_comparison(results_dir, output_dir, env_filter=None):
    """기본 학습 vs DR 학습 비교."""
    logs = discover_logs(results_dir, env_filter)
    if not logs:
        print("  [Domain Rand.] 로그 데이터가 없습니다.")
        return

    groups = defaultdict(dict)
    for log in logs:
        if log["reward"]:
            continue
        key = (log["env"], log["algo"].upper(), log["seed"])
        label = "DR" if log["dr"] else "기본"
        groups[key][label] = log

    # 기본+DR 쌍이 있는 것만
    pairs = {k: v for k, v in groups.items() if len(v) == 2}
    if not pairs:
        print("  [Domain Rand.] 비교할 데이터가 부족합니다 (기본 + DR 모두 필요).")
        return

    for (env, algo, seed), pair in pairs.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, log in pair.items():
            df = pd.read_csv(log["csv_path"])
            if "episode/return" not in df.columns:
                continue
            returns = smooth(df["episode/return"].dropna().values, window=20)
            style = "-" if label == "기본" else "--"
            ax.plot(returns, label=label, linestyle=style, linewidth=2)

        ax.set_xlabel("Log Step")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"{env} / {algo} (seed={seed}) — Domain Randomization")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, f"dr_{env}_{algo}_seed{seed}.png")
        plt.savefig(path)
        plt.close()
        print(f"  ✅ Domain Rand.: {path}")


# ─── 메인 ───

def main():
    parser = argparse.ArgumentParser(description="시각화 자동화")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--type", default="all",
                        choices=["all", "curves", "bar", "reward", "dr"])
    parser.add_argument("--env", default=None, help="특정 환경만 (예: HalfCheetah-v5)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  📈 시각화 자동화 | 타입: {args.type}")
    print(f"{'='*60}\n")

    if args.type in ("all", "curves"):
        plot_training_curves(args.results_dir, args.output_dir, args.env)
    if args.type in ("all", "bar"):
        plot_performance_bar(args.results_dir, args.output_dir)
    if args.type in ("all", "reward"):
        plot_reward_comparison(args.results_dir, args.output_dir, args.env)
    if args.type in ("all", "dr"):
        plot_dr_comparison(args.results_dir, args.output_dir, args.env)

    print(f"\n  완료! 그래프 저장 위치: {args.output_dir}/")


if __name__ == "__main__":
    main()
