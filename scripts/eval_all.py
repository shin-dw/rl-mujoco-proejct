"""
일괄 평가 스크립트 — 팀원 A (환경/실험 인프라 담당) 제작

results/ 디렉토리에서 학습 완료된 모든 모델을 자동 탐색하여 일괄 평가 후 통합 CSV 출력.

사용법:
    python scripts/eval_all.py
    python scripts/eval_all.py --episodes 30
    python scripts/eval_all.py --filter ppo
"""

import argparse, os, sys, re, csv
import numpy as np
import gymnasium as gym
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.algorithms.ppo import PPO
from src.algorithms.sac import SAC
from src.algorithms.td3 import TD3

ALGO_MAP = {"ppo": PPO, "sac": SAC, "td3": TD3}
EXP_PATTERN = re.compile(
    r"^(?P<algo>ppo|sac|td3)_(?P<env>[A-Za-z]+-v\d+)"
    r"(?:_(?P<reward>[a-z_]+?))?(?P<dr>_dr)?_seed(?P<seed>\d+)$"
)


def find_experiments(results_dir, filter_str=None):
    experiments = []
    for exp_dir in sorted(Path(results_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        if filter_str and filter_str.lower() not in exp_dir.name.lower():
            continue
        match = EXP_PATTERN.match(exp_dir.name)
        if not match:
            continue
        model_path = exp_dir / "models" / "model_final.pt"
        if not model_path.exists():
            print(f"  [건너뜀] {exp_dir.name} — model_final.pt 없음")
            continue
        info = match.groupdict()
        info["seed"] = int(info["seed"])
        info["dr"] = info["dr"] is not None
        info["name"] = exp_dir.name
        info["model_path"] = str(model_path)
        experiments.append(info)
    return experiments


def evaluate_model(algo_name, model_path, env_id, n_episodes, seed):
    env = gym.make(env_id)
    algo = ALGO_MAP[algo_name](env.observation_space.shape[0], env.action_space.shape[0])
    algo.load(model_path)

    returns, lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            action = algo.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
    env.close()

    return {
        "mean_return": np.mean(returns), "std_return": np.std(returns),
        "min_return": np.min(returns), "max_return": np.max(returns),
        "mean_length": np.mean(lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="일괄 평가")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or os.path.join(args.results_dir, "eval", "summary.csv")

    print(f"\n{'='*60}\n  📊 일괄 평가 시작 | 에피소드: {args.episodes}\n{'='*60}")
    experiments = find_experiments(args.results_dir, args.filter)
    if not experiments:
        print("[!] 평가할 실험이 없습니다. 팀원 B, C가 학습 완료 후 다시 실행하세요.")
        return

    print(f"  발견된 실험: {len(experiments)}개\n")
    rows = []
    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {exp['name']}")
        r = evaluate_model(exp["algo"], exp["model_path"], exp["env"],
                           args.episodes, exp["seed"] + 2000)
        row = {
            "experiment": exp["name"], "algorithm": exp["algo"].upper(),
            "environment": exp["env"], "seed": exp["seed"],
            "reward_type": exp.get("reward") or "default",
            "domain_rand": exp["dr"],
            "mean_return": f"{r['mean_return']:.1f}",
            "std_return": f"{r['std_return']:.1f}",
            "min_return": f"{r['min_return']:.1f}",
            "max_return": f"{r['max_return']:.1f}",
            "mean_length": f"{r['mean_length']:.0f}",
        }
        rows.append(row)
        print(f"  → Return: {r['mean_return']:.1f} ± {r['std_return']:.1f}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ 평가 완료! 결과: {output}")
    print(f"\n{'알고리즘':<8} {'환경':<18} {'시드':<6} {'Mean Return':<14} {'± Std'}")
    print("-" * 58)
    for r in rows:
        print(f"{r['algorithm']:<8} {r['environment']:<18} {r['seed']:<6} "
              f"{r['mean_return']:>12} {r['std_return']:>8}")


if __name__ == "__main__":
    main()
