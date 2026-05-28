"""
실험 자동화 스크립트

실험 구성:
  1. 베이스라인  : PPO / SAC / TD3  x  시드 42, 77, 123
  2. Reward Shaping : PPO / SAC / TD3  x  balanced_walk / stable_gait  (시드 42)
  3. HP 튜닝     : 알고리즘별 주요 HP 변경  (시드 42, 베이스라인과 비교)
  4. 최적 조합   : 1~3 결과에서 최고 설정을 자동 탐색해 최종 학습

사용법:
  python scripts/run_experiments.py                  # 전체 실행 (1~3)
  python scripts/run_experiments.py --phase baseline
  python scripts/run_experiments.py --phase reward
  python scripts/run_experiments.py --phase hp
  python scripts/run_experiments.py --phase best     # 최적 조합 (eval_all.py 먼저 실행)
  python scripts/run_experiments.py --dry-run        # 명령어만 출력 (실행 안 함)
"""

import argparse
import re
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

def build_baseline(device: str = None, algo_filter: str = None) -> list[dict]:
    algos = [algo_filter] if algo_filter else ALGOS
    experiments = []
    for algo in algos:
        for seed in SEEDS:
            exp_name = f"{algo}_{ENV}_seed{seed}"
            cmd = [
                sys.executable, "-m", "src.train",
                "--algo", algo,
                "--env", ENV,
                "--seed", str(seed),
                "--total-steps", str(TOTAL_STEPS),
                "--tensorboard",
            ]
            if device:
                cmd += ["--device", device]
            experiments.append({"name": exp_name, "cmd": cmd})
    return experiments


def run_baseline(dry_run: bool, device: str = None, algo_filter: str = None):
    print("\n" + "★" * 30)
    print("  Phase 1: 베이스라인" + (f" [{algo_filter.upper()} only]" if algo_filter else ""))
    print("★" * 30)
    experiments = build_baseline(device, algo_filter)
    print_plan(experiments)
    for exp in experiments:
        if result_exists(exp["name"]):
            print(f"  [SKIP] {exp['name']} (이미 완료됨)")
            continue
        run_cmd(exp["cmd"], dry_run)

# =============================================================================
# 실험 2: Reward Shaping
# =============================================================================

def build_reward_shaping(device: str = None, algo_filter: str = None) -> list[dict]:
    algos = [algo_filter] if algo_filter else ALGOS
    experiments = []
    for algo in algos:
        for reward_type in REWARD_TYPES:
            exp_name = f"{algo}_{ENV}_{reward_type}_seed42"
            cmd = [
                sys.executable, "-m", "src.train",
                "--algo", algo,
                "--env", ENV,
                "--seed", "42",
                "--total-steps", str(TOTAL_STEPS),
                "--reward-type", reward_type,
                "--tensorboard",
            ]
            if device:
                cmd += ["--device", device]
            experiments.append({"name": exp_name, "cmd": cmd})
    return experiments


def run_reward_shaping(dry_run: bool, device: str = None, algo_filter: str = None):
    print("\n" + "★" * 30)
    print("  Phase 2: Reward Shaping" + (f" [{algo_filter.upper()} only]" if algo_filter else ""))
    print("★" * 30)
    experiments = build_reward_shaping(device, algo_filter)
    print_plan(experiments)
    for exp in experiments:
        if result_exists(exp["name"]):
            print(f"  [SKIP] {exp['name']} (이미 완료됨)")
            continue
        run_cmd(exp["cmd"], dry_run)

# =============================================================================
# 실험 3: HP 튜닝
# =============================================================================

def build_hp_tuning(algo_filter: str = None) -> list[dict]:
    variants_map = (
        {algo_filter: HP_VARIANTS[algo_filter]}
        if algo_filter and algo_filter in HP_VARIANTS
        else HP_VARIANTS
    )
    experiments = []
    for algo, variants in variants_map.items():
        for v in variants:
            exp_name = f"{algo}_{ENV}_hp_{v['tag']}_seed42"
            experiments.append({
                "name": exp_name,
                "algo": algo,
                "variant": v,
            })
    return experiments


def run_hp_tuning(dry_run: bool, device: str = None, algo_filter: str = None):
    print("\n" + "★" * 30)
    print("  Phase 3: HP 튜닝" + (f" [{algo_filter.upper()} only]" if algo_filter else ""))
    print("★" * 30)
    experiments = build_hp_tuning(algo_filter)
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
            "--exp-name", exp["name"],
            "--tensorboard",
        ]
        if device:
            cmd += ["--device", device]
        run_cmd(cmd, dry_run)

    # 임시 config 폴더 정리
    if not dry_run and HP_CONFIG_DIR.exists():
        import shutil
        shutil.rmtree(HP_CONFIG_DIR)
        print("\n  [정리] 임시 HP config 파일 삭제 완료")

# =============================================================================
# 중간 평가: eval_all.py 실행
# =============================================================================

def run_eval_all(dry_run: bool):
    print("\n" + "★" * 30)
    print("  중간 평가: eval_all.py")
    print("★" * 30)
    cmd = [sys.executable, "scripts/eval_all.py", "--episodes", "20"]
    run_cmd(cmd, dry_run)


def run_report(dry_run: bool):
    print("\n" + "★" * 30)
    print("  보고서 생성: generate_report.py")
    print("★" * 30)
    # 그래프 먼저 생성
    run_cmd([sys.executable, "scripts/plot_results.py", "--env", "Humanoid-v5"], dry_run)
    # 보고서 생성
    run_cmd([sys.executable, "scripts/generate_report.py"], dry_run)


# =============================================================================
# 실험 4: 최적 조합
# =============================================================================

def find_best_settings(results_dir: str = "results") -> dict:
    """
    summary.csv에서 최고 성능 설정을 자동으로 탐색합니다.
    반환: {algo, reward, hp_tag, hp_variant}
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas가 필요합니다: pip install pandas")

    summary_path = Path(results_dir) / "eval" / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "summary.csv가 없습니다.\n"
            "  먼저 다음 명령어를 실행하세요:\n"
            "  python scripts/eval_all.py"
        )

    df = pd.read_csv(summary_path)
    df["mean_return"] = pd.to_numeric(df["mean_return"], errors="coerce")

    # 1. 최고 알고리즘: 베이스라인 시드 평균 기준
    baseline_mask = df["experiment"].str.match(
        rf"^(ppo|sac|td3)_{re.escape(ENV)}_seed\d+$"
    )
    baseline = df[baseline_mask]
    if baseline.empty:
        raise ValueError("베이스라인 평가 결과가 없습니다. 베이스라인을 먼저 실행하세요.")
    best_algo = baseline.groupby("algorithm")["mean_return"].mean().idxmax().lower()

    # 2. 최고 Reward Shaping: 해당 알고리즘의 reward 실험 중 최고
    reward_mask = df["experiment"].str.match(
        rf"^{best_algo}_{re.escape(ENV)}_({'|'.join(REWARD_TYPES)})_seed42$"
    )
    reward_df = df[reward_mask]
    best_reward = None
    if not reward_df.empty:
        best_reward_exp = reward_df.loc[reward_df["mean_return"].idxmax(), "experiment"]
        m = re.search(r"(" + "|".join(REWARD_TYPES) + r")", best_reward_exp)
        if m:
            # 베이스라인보다 높을 때만 채택
            baseline_mean = baseline[baseline["algorithm"] == best_algo.upper()]["mean_return"].mean()
            if reward_df["mean_return"].max() > baseline_mean:
                best_reward = m.group(1)

    # 3. 최고 HP: 해당 알고리즘의 HP 실험 중 최고
    hp_mask = df["experiment"].str.match(
        rf"^{best_algo}_{re.escape(ENV)}_hp_[a-z0-9e.\-]+_seed42$"
    )
    hp_df = df[hp_mask]
    best_hp_tag = None
    best_hp_variant = None
    if not hp_df.empty:
        best_hp_exp = hp_df.loc[hp_df["mean_return"].idxmax(), "experiment"]
        m = re.search(r"_hp_([a-z0-9e.\-]+)_seed", best_hp_exp)
        if m:
            tag = m.group(1)
            baseline_mean = baseline[baseline["algorithm"] == best_algo.upper()]["mean_return"].mean()
            if hp_df["mean_return"].max() > baseline_mean:
                best_hp_tag = tag
                for v in HP_VARIANTS.get(best_algo, []):
                    if v["tag"] == tag:
                        best_hp_variant = v
                        break

    return {
        "algo": best_algo,
        "reward": best_reward,
        "hp_tag": best_hp_tag,
        "hp_variant": best_hp_variant,
    }


def run_best_combination(dry_run: bool, device: str = None):
    print("\n" + "★" * 30)
    print("  Phase 4: 최적 조합")
    print("★" * 30)

    try:
        best = find_best_settings()
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"\n  [오류] {e}")
        return

    algo      = best["algo"]
    reward    = best["reward"]
    hp_tag    = best["hp_tag"]
    hp_variant = best["hp_variant"]

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  최고 알고리즘 : {algo.upper():<17}│")
    print(f"  │  최고 Reward   : {(reward or '기본 (개선 없음)'):<17}│")
    print(f"  │  최고 HP       : {(hp_tag or '기본 (개선 없음)'):<17}│")
    print(f"  └─────────────────────────────────┘")

    # HP config 생성
    config_path = BASE_CONFIG
    if hp_variant:
        HP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = make_hp_config(
            algo=algo,
            section=hp_variant["section"],
            key=hp_variant["key"],
            value=hp_variant["value"],
            tag=f"best_{hp_tag}",
        )

    reward_suffix = f"_{reward}" if reward else ""
    hp_suffix     = f"_hp_{hp_tag}" if hp_tag else ""
    exp_name = f"{algo}_{ENV}{reward_suffix}{hp_suffix}_best_seed42"

    cmd = [
        sys.executable, "-m", "src.train",
        "--algo", algo,
        "--env", ENV,
        "--seed", "42",
        "--total-steps", str(TOTAL_STEPS),
        "--config", str(config_path),
        "--tensorboard",
    ]
    if reward:
        cmd += ["--reward-type", reward]
    if device:
        cmd += ["--device", device]

    print(f"\n  실험명: {exp_name}\n")

    if result_exists(exp_name):
        print(f"  [SKIP] {exp_name} (이미 완료됨)")
        return

    run_cmd(cmd, dry_run)

    if not dry_run and hp_variant and HP_CONFIG_DIR.exists():
        import shutil
        shutil.rmtree(HP_CONFIG_DIR)


# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="희승 실험 자동화 스크립트")
    parser.add_argument(
        "--phase",
        choices=["all", "baseline", "reward", "hp", "best"],
        default="all",
        help="실행할 실험 단계 (기본: all = 1~3단계, best는 별도 실행)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="명령어만 출력하고 실제로 실행하지 않음",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="학습 디바이스 (기본: config 값 사용, cpu 강제 지정 가능)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["ppo", "sac", "td3"],
        help="특정 알고리즘만 실행 (기본: 전체 알고리즘)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("\n  [DRY-RUN 모드] 명령어만 출력합니다.\n")
    if args.device:
        print(f"\n  [디바이스 오버라이드] --device {args.device}")
    if args.algo:
        print(f"\n  [알고리즘 필터] {args.algo.upper()} 만 실행")

    total = (
        len(build_baseline(algo_filter=args.algo)) +
        len(build_reward_shaping(algo_filter=args.algo)) +
        len(build_hp_tuning(algo_filter=args.algo))
    )
    print(f"\n  환경: {ENV} | 총 예정 실험: {total}개")
    print(f"  스텝: {TOTAL_STEPS:,} / 실험")

    if args.phase in ("all", "baseline"):
        run_baseline(args.dry_run, args.device, args.algo)

    if args.phase in ("all", "reward"):
        run_reward_shaping(args.dry_run, args.device, args.algo)

    if args.phase in ("all", "hp"):
        run_hp_tuning(args.dry_run, args.device, args.algo)

    if args.phase == "all":
        run_eval_all(args.dry_run)
        run_best_combination(args.dry_run, args.device)
        run_eval_all(args.dry_run)   # 최적 조합 포함 재평가
        run_report(args.dry_run)

    if args.phase == "best":
        run_best_combination(args.dry_run, args.device)

    print("\n\n  모든 실험 완료!")


if __name__ == "__main__":
    main()
