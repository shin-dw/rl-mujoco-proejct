"""
통합 학습 스크립트

사용 예시:
    python -m src.train --algo ppo --env HalfCheetah-v5 --seed 42
    python -m src.train --algo sac --env Ant-v5 --total-steps 2000000
"""

import argparse
import os
import yaml
import numpy as np
import torch
from typing import Dict, Any

from .common.env_wrapper import make_env
from .common.logger import Logger
from .common.evaluator import Evaluator, _find_normalize_wrapper
from .rewards.custom_rewards import get_reward_fn
from .algorithms.ppo import PPO
from .algorithms.sac import SAC
from .algorithms.td3 import TD3


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo RL 학습 스크립트")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=None)
    parser.add_argument("--reward-type", type=str, default=None)
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--normalize-obs", action="store_true", default=True)
    parser.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="실험 폴더명 직접 지정 (기본: 자동 생성)")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"],
                        help="학습 디바이스 (기본: config 값 사용)")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_algorithm(name, obs_dim, act_dim, config, device="auto", seed=42, max_action=1.0):
    net = config.get("network", {})
    h = net.get("hidden_dims", [256, 256])
    act = net.get("activation", "relu")
    gamma = config.get("common", {}).get("gamma", 0.99)
    c = config.get(name, {})

    if name == "ppo":
        return PPO(obs_dim, act_dim, h, act, c.get("lr_actor", 3e-4), c.get("lr_critic", 3e-4),
                    gamma, c.get("gae_lambda", 0.95), c.get("clip_ratio", 0.2),
                    c.get("n_epochs", 10), c.get("batch_size", 64), c.get("n_steps", 2048),
                    c.get("entropy_coef", 0.0), c.get("value_loss_coef", 0.5),
                    c.get("max_grad_norm", 0.5), c.get("normalize_advantage", True),
                    c.get("lr_annealing", False), device, seed)
    elif name == "sac":
        return SAC(obs_dim, act_dim, h, act, c.get("lr_actor", 3e-4), c.get("lr_critic", 3e-4),
                    c.get("lr_alpha", 3e-4), gamma, c.get("tau", 0.005), c.get("batch_size", 256),
                    c.get("buffer_size", 1000000), c.get("learning_starts", 10000),
                    c.get("auto_entropy", True), c.get("init_alpha", 0.2),
                    c.get("target_entropy", None), max_action, device, seed,
                    gradient_steps=c.get("gradient_steps", 1))
    elif name == "td3":
        return TD3(obs_dim, act_dim, h, act, c.get("lr_actor", 3e-4), c.get("lr_critic", 3e-4),
                    gamma, c.get("tau", 0.005), c.get("batch_size", 256),
                    c.get("buffer_size", 1000000), c.get("learning_starts", 10000),
                    c.get("policy_delay", 2), c.get("exploration_noise", 0.1),
                    c.get("target_noise", 0.2), c.get("noise_clip", 0.5), max_action, device, seed,
                    gradient_steps=c.get("gradient_steps", 1))
    raise ValueError(f"지원하지 않는 알고리즘: {name}")


def save_with_obs_stats(algo, env, path: str):
    """모델 저장 시 NormalizeObservation 통계를 함께 보관합니다.

    모델 파일과 동일한 경로에 _obs_stats.npz로 저장하여
    evaluate.py 단독 실행 시에도 정확한 정규화를 복원할 수 있습니다.
    """
    algo.save(path)
    norm = _find_normalize_wrapper(env)
    if norm is not None:
        stats_path = path.replace(".pt", "_obs_stats.npz")
        np.savez(
            stats_path,
            running_mean=norm.running_mean,
            running_var=norm.running_var,
            count=np.array([norm.count]),
        )


def train_ppo(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir):
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    for step in range(1, total_steps + 1):
        # LR 선형 감소 (lr_annealing=False이면 no-op)
        algo.anneal_lr(step, total_steps)

        action, log_prob, value = algo.select_action(obs)
        # NOTE: 버퍼에는 원본 가우시안 샘플을 그대로 저장.
        # MuJoCo가 내부적으로 ctrlrange(±0.4)로 클리핑하므로 env.step에는 원본을 넘겨도 무방.
        # "클리핑 후 저장" 방식은 경계값 액션이 log_std 그래디언트를 양수 방향으로 편향시켜
        # entropy 폭발을 유발하므로 표준 PPO 방식(원본 저장)을 따름.
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # [FIX 2] GAE 부트스트래핑을 위해 terminated만 버퍼에 기록.
        # truncated(타임아웃)를 done=1로 저장하면 GAE가 next_value를 무시(×0)해
        # 긴 에피소드의 리턴을 과소추정함. terminated만 진짜 terminal state임.
        algo.buffer.add(obs, action, reward, value, log_prob, terminated)
        ep_ret += reward
        ep_len += 1
        obs = next_obs

        if done:
            logger.log_episode(ep_ret, ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if algo.buffer.full:
            with torch.no_grad():
                last_val = algo.critic(algo._to_tensor(obs).unsqueeze(0)).item()
            metrics = algo.update(last_val, done)
            for k, v in metrics.items():
                logger.log_scalar(k, v, step)

        if step % log_freq == 0:
            if algo.lr_annealing:
                logger.log_scalar("train/lr", algo.actor_optimizer.param_groups[0]["lr"], step)
            logger.dump(step)
        if step % eval_freq == 0:
            r = evaluator.evaluate(algo.select_action, step)
            for k, v in r.items():
                logger.log_scalar(k, v, step)
            print(f"  [평가] Return: {r['eval/mean_return']:.1f} ± {r['eval/std_return']:.1f}")
        if step % (eval_freq * 5) == 0:
            algo.total_steps = step
            save_with_obs_stats(algo, env, os.path.join(save_dir, "models", f"model_{step}.pt"))

    algo.total_steps = total_steps
    save_with_obs_stats(algo, env, os.path.join(save_dir, "models", "model_final.pt"))


def train_offpolicy(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir,
                    update_freq: int = 1):
    """
    Off-policy 학습 루프.

    update_freq  : 몇 스텝마다 한 번 업데이트할지 (기본 1 = 매 스텝)
    gradient_steps는 algo 생성자에서 설정되어 algo.update() 내부에서 처리됨.
    → Python 루프 오버헤드 최소화 + GPU-CPU 동기화(.item())를 gradient_steps에 무관하게 3~4회로 고정
    """
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    last_metrics: dict = {}

    for step in range(1, total_steps + 1):
        if step < algo.learning_starts:
            action = env.action_space.sample()
        else:
            action = algo.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        algo.buffer.add(obs, action, reward, next_obs, terminated)
        ep_ret += reward
        ep_len += 1
        obs = next_obs

        if done:
            logger.log_episode(ep_ret, ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        # update_freq 스텝마다 1회 업데이트 (gradient_steps는 algo 내부에서 처리)
        if step >= algo.learning_starts and step % update_freq == 0:
            last_metrics = algo.update()
            for k, v in last_metrics.items():
                logger.log_scalar(k, v, step)

        if step % log_freq == 0:
            logger.dump(step)
        if step % eval_freq == 0:
            r = evaluator.evaluate(algo.select_action, step)
            for k, v in r.items():
                logger.log_scalar(k, v, step)
            print(f"  [평가] Return: {r['eval/mean_return']:.1f} ± {r['eval/std_return']:.1f}")
        if step % (eval_freq * 5) == 0:
            algo.total_steps = step
            save_with_obs_stats(algo, env, os.path.join(save_dir, "models", f"model_{step}.pt"))

    algo.total_steps = total_steps
    save_with_obs_stats(algo, env, os.path.join(save_dir, "models", "model_final.pt"))


def main():
    args = parse_args()
    config = load_config(args.config)
    cc = config.get("common", {})

    env_ov = config.get("env_overrides", {}).get(args.env, {})
    total_steps = args.total_steps or env_ov.get("total_timesteps", cc.get("total_timesteps", 1000000))
    eval_freq = args.eval_freq or cc.get("eval_freq", 10000)
    log_freq = args.log_freq or cc.get("log_freq", 1000)

    rs = f"_{args.reward_type}" if args.reward_type else ""
    ds = "_dr" if args.domain_rand else ""
    exp_name = args.exp_name or f"{args.algo}_{args.env}{rs}{ds}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    reward_fn = get_reward_fn(args.env, args.reward_type) if args.reward_type else None
    dr_cfg = config.get("domain_randomization", {}) if args.domain_rand else None
    # SAC·TD3: normalize_reward는 replay buffer와 비호환 → 대신 reward_scale(고정 나눗셈) 사용
    # PPO: normalize_reward=False, reward_scale=10.0 (run_forward 리워드 크기 안정화)
    normalize_reward = config.get(args.algo, {}).get("normalize_reward", False)
    reward_scale = config.get(args.algo, {}).get("reward_scale", 1.0)

    env = make_env(args.env, args.seed, args.normalize_obs, normalize_reward=normalize_reward,
                   gamma=cc.get("gamma", 0.99),
                   domain_randomization=args.domain_rand, dr_config=dr_cfg, custom_reward_fn=reward_fn,
                   reward_scale=reward_scale)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # 환경의 실제 action bound를 읽어 TD3 max_action에 전달
    max_action = float(env.action_space.high[0])
    device = args.device or cc.get("device", "auto")

    if device == "cpu":
        device_info = "CPU (강제 지정)"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU가 요청됐지만 CUDA를 사용할 수 없습니다.\n"
                "  - Docker: GPU 빌드 이미지 사용 여부 확인 (docker-compose train-single-gpu)\n"
                "  - 로컬:   pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device == "auto" and torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_info = "CPU"
    print(f"{'='*60}\n  알고리즘: {args.algo.upper()} | 환경: {args.env} | 시드: {args.seed}")
    print(f"  총 스텝: {total_steps:,} | 저장: {save_dir} | 디바이스: {device_info}\n{'='*60}")
    algo = create_algorithm(args.algo, obs_dim, act_dim, config, device, args.seed, max_action)
    if args.load_model:
        algo.load(args.load_model)

    logger = Logger(os.path.join(save_dir, "logs"), exp_name, args.tensorboard)
    evaluator = Evaluator(args.env, cc.get("eval_episodes", 10), args.seed + 1000, train_env=env)

    try:
        if args.algo == "ppo":
            train_ppo(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir)
        else:
            ac = config.get(args.algo, {})
            update_freq    = ac.get("update_freq", 1)
            gradient_steps = ac.get("gradient_steps", 1)
            print(f"  update_freq={update_freq} | gradient_steps={gradient_steps} (algo 내부 처리)")
            train_offpolicy(algo, env, evaluator, logger, total_steps, eval_freq, log_freq,
                            save_dir, update_freq=update_freq)
    finally:
        logger.close()
        evaluator.close()
        env.close()
    print("\n학습 완료!")


if __name__ == "__main__":
    main()
