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
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.algorithms.ppo import PPO
from src.algorithms.sac import SAC
from src.algorithms.td3 import TD3
# 학습 시와 동일한 커스텀 래퍼 사용 (gymnasium 내장 NormalizeObservation과 다름)
from src.common.env_wrapper import NormalizeObservation as ProjectNormalizeObs

ALGO_MAP = {"ppo": PPO, "sac": SAC, "td3": TD3}


def infer_hidden_dims(model_path: str, algo_name: str):
    """체크포인트 weight shape에서 hidden_dims를 자동 추론."""
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        actor = state.get("actor", {})

        if algo_name in ("sac", "ppo"):
            # GaussianActor: shared.0, shared.2, ... 이 모두 hidden layer
            dims, i = [], 0
            while f"shared.{i * 2}.weight" in actor:
                dims.append(actor[f"shared.{i * 2}.weight"].shape[0])
                i += 1
            if dims:
                return dims

        elif algo_name == "td3":
            # DeterministicActor: net.net.0, .2, .4 ... 마지막은 출력층
            dims, i = [], 0
            while f"net.net.{i * 2}.weight" in actor:
                dims.append(actor[f"net.net.{i * 2}.weight"].shape[0])
                i += 1
            if len(dims) > 1:
                return dims[:-1]   # 마지막 = act_dim 출력층 제외

    except Exception as e:
        print(f"  [경고] 아키텍처 추론 실패 ({e}), 기본값 [256,256] 사용")

    return [256, 256]
EXP_PATTERN = re.compile(
    r"^(?P<algo>ppo|sac|td3)_(?P<env>[A-Za-z]+-v\d+)"
    r"(?:_(?P<reward>[a-z][a-z0-9_-]+?))?"   # reward type 또는 hp_* 태그 (숫자·하이픈 허용)
    r"(?P<dr>_dr)?_seed(?P<seed>\d+)$"
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
    # 프로젝트 전용 NormalizeObservation 사용 — 학습 코드와 동일한 구현
    env = ProjectNormalizeObs(gym.make(env_id))

    # 학습 중 누적된 running stats 복원
    stats_path = model_path.replace(".pt", "_obs_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        env.running_mean = stats["running_mean"].copy()
        env.running_var  = stats["running_var"].copy()
        env.count        = float(stats["count"][0])
        print(f"  obs_stats 복원 완료 (count={env.count:.0f})")
    else:
        print(f"  [경고] obs_stats 없음 — 정규화 통계 초기화 상태로 평가")

    # 평가 중 stats 업데이트 비활성화 — 복원한 통계가 오염되지 않도록 freeze
    env._update_stats = lambda obs: None

    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    max_action = float(env.env.action_space.high[0])   # 원본 env에서 읽기

    # 체크포인트에서 실제 학습에 사용된 hidden_dims 추론
    hidden_dims = infer_hidden_dims(model_path, algo_name)
    print(f"  hidden_dims={hidden_dims}, max_action={max_action:.2f}")

    if algo_name == "ppo":
        algo = PPO(obs_dim, act_dim, hidden_dims=hidden_dims)
    elif algo_name == "sac":
        algo = SAC(obs_dim, act_dim, hidden_dims=hidden_dims, max_action=max_action)
    elif algo_name == "td3":
        algo = TD3(obs_dim, act_dim, hidden_dims=hidden_dims, max_action=max_action)
    else:
        raise ValueError(f"지원하지 않는 알고리즘: {algo_name}")

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
        reward_val = exp.get("reward") or ""
        is_hp = reward_val.startswith("hp_")
        row = {
            "experiment": exp["name"], "algorithm": exp["algo"].upper(),
            "environment": exp["env"], "seed": exp["seed"],
            "exp_type": "hp_tuning" if is_hp else ("reward_shaping" if reward_val else "baseline"),
            "reward_type": "" if is_hp else (reward_val or "default"),
            "hp_tag": reward_val[3:] if is_hp else "",   # hp_ 접두사 제거
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
