"""
실험 자동화 스크립트

실험 구성:
  1. 베이스라인  : PPO / SAC / TD3  x  시드 42, 77, 123
  2. Reward Shaping : PPO / SAC / TD3  x  balanced_walk / stable_gait  (시드 42)
  3. HP 튜닝     : 알고리즘별 주요 HP 변경  (시드 42, 베이스라인과 비교)

사용법:
  python scripts/run_experiments.py                  # 전체 실행
  python scripts/run_experiments.py --phase baseline
  python scripts/run_experiments.py --phase reward
  python scripts/run_experiments.py --phase hp
  python scripts/run_experiments.py --dry-run        # 명령어만 출력 (실행 안 함)
"""

import argparse
import os
import subprocess
import sys
import yaml
import copy
from pathlib import Path

# =============================================================================
# 실험 설정
# =============================================================================

ENV          = "Humanoid-v5"
SEEDS        = [42, 77, 123]
ALGOS        = ["ppo", "sac", "td3"]
REWARD_TYPES = ["balanced_walk", "stable_gait"]
TOTAL_STEPS  = 3_000_000
BASE_CONFIG  = "configs/default.yaml"
HP_CONFIG_DIR = Path("configs/hp_tuning")

# HP 튜닝 변형 정의 (기본값 제외한 비교 대상만)
HP_VARIANTS = {
    "ppo": [
        {"tag": "clip01",  "section": "ppo", "key": "clip_ratio",  "value": 0.1},
        {"tag": "clip03",  "section": "ppo", "key": "clip_ratio",  "value": 0.3},
        {"tag": "lr1e-4",  "section": "ppo", "key": "lr_actor",    "value": 1e-4},
        {"tag": "lr1e-3",  "section": "ppo", "key": "lr_actor",    "value": 1e-3},
    ],
    "sac": [
        {"tag": "lra1e-4", "section": "sac", "key": "lr_alpha",   "value": 1e-4},
        {"tag": "lra1e-3", "section": "sac", "key": "lr_alpha",   "value": 1e-3},
        {"tag": "bs128",   "section": "sac", "key": "batch_size", "value": 128},
        {"tag": "bs512",   "section": "sac", "key": "batch_size", "value": 512},
    ],
    "td3": [
        {"tag": "delay1",    "section": "td3", "key": "policy_delay",      "value": 1},
        {"tag": "delay4",    "section": "td3", "key": "policy_delay",      "value": 4},
        {"tag": "noise005",  "section": "td3", "key": "exploration_noise", "value": 0.05},
        {"tag": "noise02",   "section": "td3", "key": "exploration_noise", "value": 0.2},
    ],
}

# =============================================================================
# 유틸리티
# =============================================================================

def result_exists(exp_name: str) -> bool:
    """이미 완료된 실험인지 확인 (결과 폴더 존재 여부)."""
    path = Path("results") / exp_name / "logs" / "progress.csv"
    return path.exists()


def run_cmd(cmd: list[str], dry_run: bool):
    """명령어 실행 (dry_run이면 출력만)."""
    print("\n" + "=" * 70)
    print("  $ " + " ".join(cmd))
    print("=" * 70)
    if not dry_run:
        subprocess.run(cmd, check=True)


def make_hp_config(algo: str, section: str, key: str, value, tag: str) -> Path:
    """기본 config를 복사해 HP 하나만 변경한 임시 config 파일 생성."""
    HP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_copy = copy.deepcopy(cfg)
    cfg_copy[section][key] = value

    out_path = HP_CONFIG_DIR / f"{algo}_{tag}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_copy, f, allow_unicode=True)
    return out_path


def print_plan(experiments: list[dict]):
    """실행 예정 실험 목록을 표로 출력."""
    print(f"\n{'─'*70}")
    print(f"  총 실험 수: {len(experiments)}개")
    print(f"{'─'*70}")
    for i, exp in enumerate(experiments, 1):
        print(f"  [{i:>2}] {exp['name']}")
    print(f"{'─'*70}\n")

# =============================================================================
# 실험 1: 베이스라인
# =============================================================================

def build_baseline() -> list[dict]:
    experiments = []
    for algo in ALGOS:
        for seed in SEEDS:
            exp_name = f"{algo}_{ENV}_seed{seed}"
            experiments.append({
                "name": exp_name,
                "cmd": [
                    sys.executable, "-m", "src.train",
                    "--algo", algo,
                    "--env", ENV,
                    "--seed", str(seed),
                    "--total-steps", str(TOTAL_STEPS),
                    "--tensorboard",
                ],
            })
    return experiments


def run_baseline(dry_run: bool):
    print("\n" + "★" * 30)
    print("  Phase 1: 베이스라인")
    print("★" * 30)
    experiments = build_baseline()
    print_plan(experiments)
    for exp in experiments:
        if result_exists(exp["name"]):
            print(f"  [SKIP] {exp['name']} (이미 완료됨)")
            continue
        run_cmd(exp["cmd"], dry_run)

# =============================================================================
# 실험 2: Reward Shaping
# =============================================================================

def build_reward_shaping() -> list[dict]:
    experiments = []
    for algo in ALGOS:
        for reward_type in REWARD_TYPES:
            exp_name = f"{algo}_{ENV}_{reward_type}_seed42"
            experiments.append({
                "name": exp_name,
                "cmd": [
                    sys.executable, "-m", "src.train",
                    "--algo", algo,
                    "--env", ENV,
                    "--seed", "42",
                    "--total-steps", str(TOTAL_STEPS),
                    "--reward-type", reward_type,
                    "--tensorboard",
                ],
            })
    return experiments


def run_reward_shaping(dry_run: bool):
    print("\n" + "★" * 30)
    print("  Phase 2: Reward Shaping")
    print("★" * 30)
    experiments = build_reward_shaping()
    print_plan(experiments)
    for exp in experiments:
        if result_exists(exp["name"]):
            print(f"  [SKIP] {exp['name']} (이미 완료됨)")
            continue
        run_cmd(exp["cmd"], dry_run)

# =============================================================================
# 실험 3: HP 튜닝
# =============================================================================

def build_hp_tuning() -> list[dict]:
    experiments = []
    for algo, variants in HP_VARIANTS.items():
        for v in variants:
            exp_name = f"{algo}_{ENV}_hp_{v['tag']}_seed42"
            experiments.append({
                "name": exp_name,
                "algo": algo,
                "variant": v,
            })
    return experiments


def run_hp_tuning(dry_run: bool):
    print("\n" + "★" * 30)
    print("  Phase 3: HP 튜닝")
    print("★" * 30)
    experiments = build_hp_tuning()
    print_plan(experiments)

    for exp in experiments:
        if result_exists(exp["name"]):
            print(f"  [SKIP] {exp['name']} (이미 완료됨)")
            continue

        v = exp["variant"]
        config_path = make_hp_config(
            algo=exp["algo"],
            section=v["section"],
            key=v["key"],
            value=v["value"],
            tag=v["tag"],
        )

        cmd = [
            sys.executable, "-m", "src.train",
            "--algo", exp["algo"],
            "--env", ENV,
            "--seed", "42",
            "--total-steps", str(TOTAL_STEPS),
            "--config", str(config_path),
            "--save-dir", f"results",
            "--tensorboard",
        ]
        run_cmd(cmd, dry_run)

    # 임시 config 폴더 정리
    if not dry_run and HP_CONFIG_DIR.exists():
        import shutil
        shutil.rmtree(HP_CONFIG_DIR)
        print("\n  [정리] 임시 HP config 파일 삭제 완료")

# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="희승 실험 자동화 스크립트")
    parser.add_argument(
        "--phase",
        choices=["all", "baseline", "reward", "hp"],
        default="all",
        help="실행할 실험 단계 (기본: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="명령어만 출력하고 실제로 실행하지 않음",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("\n  [DRY-RUN 모드] 명령어만 출력합니다.\n")

    total = (
        len(build_baseline()) +
        len(build_reward_shaping()) +
        len(build_hp_tuning())
    )
    print(f"\n  환경: {ENV} | 총 예정 실험: {total}개")
    print(f"  스텝: {TOTAL_STEPS:,} / 실험")

    if args.phase in ("all", "baseline"):
        run_baseline(args.dry_run)

    if args.phase in ("all", "reward"):
        run_reward_shaping(args.dry_run)

    if args.phase in ("all", "hp"):
        run_hp_tuning(args.dry_run)

    print("\n\n  모든 실험 완료!")


if __name__ == "__main__":
    main()
