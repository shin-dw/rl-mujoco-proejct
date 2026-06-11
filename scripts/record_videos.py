"""
4개 실험 에피소드 녹화 스크립트

대상 실험:
  - ppo_Humanoid-v5_seed42
  - sac_Humanoid-v5_seed42
  - td3_Humanoid-v5_seed42
  - ppo_Humanoid-v5_run_forward_seed42

각 실험의 model_final.pt를 로드하여 에피소드를 실행하고
모든 에피소드 프레임을 하나로 이어붙인 MP4 영상으로 저장합니다.
(실험당 파일 1개: results/videos/{실험명}.mp4)

사전 설치:
    pip install imageio imageio-ffmpeg

사용법:
    python scripts/record_videos.py
    python scripts/record_videos.py --episodes 3 --output results/videos
    python scripts/record_videos.py --filter sac   # 특정 실험만
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from src.common.env_wrapper import NormalizeObservation as ProjectNormalizeObs
from src.algorithms.ppo import PPO
from src.algorithms.sac import SAC
from src.algorithms.td3 import TD3

# =============================================================================
# 실험 목록
# =============================================================================

EXPERIMENTS = [
    {"name": "ppo_Humanoid-v5_seed42",              "algo": "ppo", "env": "Humanoid-v5"},
    {"name": "sac_Humanoid-v5_seed42",              "algo": "sac", "env": "Humanoid-v5"},
    {"name": "td3_Humanoid-v5_seed42",              "algo": "td3", "env": "Humanoid-v5"},
    {"name": "ppo_Humanoid-v5_run_forward_seed42",  "algo": "ppo", "env": "Humanoid-v5"},
]

# =============================================================================
# 유틸리티
# =============================================================================

def infer_hidden_dims(model_path: str, algo_name: str):
    """체크포인트 가중치 형상에서 hidden_dims 자동 추론."""
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    actor = state.get("actor", {})

    if algo_name in ("ppo", "sac"):
        dims, i = [], 0
        while f"shared.{i * 2}.weight" in actor:
            dims.append(actor[f"shared.{i * 2}.weight"].shape[0])
            i += 1
        if dims:
            return dims

    elif algo_name == "td3":
        dims, i = [], 0
        while f"net.net.{i * 2}.weight" in actor:
            dims.append(actor[f"net.net.{i * 2}.weight"].shape[0])
            i += 1
        if len(dims) > 1:
            return dims[:-1]

    return [256, 256]


def build_algo(algo_name: str, obs_dim: int, act_dim: int,
               hidden_dims: list, max_action: float):
    """알고리즘 인스턴스 생성 (평가용, device=cpu)."""
    if algo_name == "ppo":
        return PPO(obs_dim, act_dim, hidden_dims=hidden_dims, device="cpu")
    elif algo_name == "sac":
        return SAC(obs_dim, act_dim, hidden_dims=hidden_dims,
                   max_action=max_action, device="cpu")
    elif algo_name == "td3":
        return TD3(obs_dim, act_dim, hidden_dims=hidden_dims,
                   max_action=max_action, device="cpu")
    raise ValueError(f"지원하지 않는 알고리즘: {algo_name}")


def save_video(frames: list, path: str, fps: int = 60):
    """imageio로 MP4 저장. ffmpeg 없으면 GIF 폴백."""
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, quality=8)
        return path
    except Exception as mp4_err:
        gif_path = path.replace(".mp4", ".gif")
        try:
            import imageio
            imageio.mimsave(gif_path, [f[::2, ::2] for f in frames], fps=fps // 2)
            print(f"    (MP4 실패: {mp4_err}  →  GIF로 저장)")
            return gif_path
        except Exception as gif_err:
            print(f"    [오류] 영상 저장 실패: {gif_err}")
            return None


# =============================================================================
# 단일 실험 녹화
# =============================================================================

def record_experiment(exp: dict, results_dir: str, output_dir: str,
                      n_episodes: int, seed: int, fps: int):
    """
    model_final.pt를 로드하고 n_episodes 에피소드를 녹화합니다.
    에피소드별로 개별 MP4로 저장합니다.
    저장 경로: {output_dir}/{실험명}/ep01.mp4, ep02.mp4, ...
    """
    results_path = Path(results_dir)
    model_path   = results_path / exp["name"] / "models" / "model_final.pt"
    stats_path   = str(model_path).replace(".pt", "_obs_stats.npz")

    if not model_path.exists():
        print(f"  [건너뜀] 모델 없음: {model_path}")
        return

    video_dir = Path(output_dir) / exp["name"]
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ── {exp['name']} ──")
    print(f"     모델: {model_path}")

    # ── 환경 생성 (rgb_array 모드로 프레임 캡처) ─────────────
    base_env = gym.make(exp["env"], render_mode="rgb_array")
    env = ProjectNormalizeObs(base_env)

    # obs 통계 복원 & 동결
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        env.running_mean = stats["running_mean"].copy()
        env.running_var  = stats["running_var"].copy()
        env.count        = float(stats["count"][0])
        env._update_stats = lambda obs: None   # 평가 중 통계 업데이트 금지
        print(f"     obs_stats 복원 완료 (count={env.count:.0f})")
    else:
        print(f"     [주의] obs_stats 없음 — 정규화 미적용")

    # ── 알고리즘 로드 ──────────────────────────────────────────
    obs_dim     = env.observation_space.shape[0]
    act_dim     = env.action_space.shape[0]
    max_action  = float(env.action_space.high[0])
    hidden_dims = infer_hidden_dims(str(model_path), exp["algo"])

    algo = build_algo(exp["algo"], obs_dim, act_dim, hidden_dims, max_action)
    algo.load(str(model_path))
    print(f"     hidden_dims: {hidden_dims} | obs_dim: {obs_dim} | act_dim: {act_dim}")

    # ── 에피소드별 개별 녹화 ──────────────────────────────────
    for ep in range(n_episodes):
        frames = []
        obs, _ = env.reset(seed=seed + ep)

        # 첫 프레임 캡처
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        ep_ret, ep_len, done = 0.0, 0, False
        while not done:
            action = algo.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated

            frame = env.render()
            if frame is not None:
                frames.append(frame)

        video_path = str(video_dir / f"ep{ep + 1:02d}.mp4")
        saved = save_video(frames, video_path, fps=fps)
        if saved:
            print(f"     ep{ep + 1:02d} → Return: {ep_ret:7.1f} | Length: {ep_len:4d}"
                  f" | 프레임: {len(frames)} | 저장: {Path(saved).name}")

    env.close()


# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="4개 실험 에피소드 녹화")
    parser.add_argument("--results-dir", default="results/models",
                        help="실험 결과 루트 디렉터리")
    parser.add_argument("--output",      default="results/videos",
                        help="영상 저장 디렉터리")
    parser.add_argument("--episodes",    type=int, default=10,
                        help="실험당 녹화할 에피소드 수 — 모두 이어붙여 파일 1개로 저장 (기본: 10)")
    parser.add_argument("--seed",        type=int, default=0,
                        help="에피소드 초기 시드 (기본: 0)")
    parser.add_argument("--fps",         type=int, default=60,
                        help="영상 FPS (기본: 60)")
    parser.add_argument("--filter",      default=None,
                        help="실험명 필터 (예: sac, ppo, balanced)")
    args = parser.parse_args()

    # imageio 설치 확인
    try:
        import imageio  # noqa
    except ImportError:
        print("[오류] imageio가 설치되어 있지 않습니다.")
        print("  pip install imageio imageio-ffmpeg")
        return

    targets = EXPERIMENTS
    if args.filter:
        targets = [e for e in EXPERIMENTS if args.filter.lower() in e["name"].lower()]
        if not targets:
            print(f"[오류] --filter '{args.filter}'에 해당하는 실험이 없습니다.")
            return

    print(f"\n{'='*60}")
    print(f"  에피소드 녹화 | {len(targets)}개 실험 × {args.episodes}에피소드")
    print(f"  저장 위치: {args.output}/")
    print(f"{'='*60}")

    for exp in targets:
        record_experiment(
            exp,
            results_dir=args.results_dir,
            output_dir=args.output,
            n_episodes=args.episodes,
            seed=args.seed,
            fps=args.fps,
        )

    print(f"\n{'='*60}")
    print(f"  녹화 완료! → {args.output}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
