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
from .common.evaluator import Evaluator
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
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_algorithm(name, obs_dim, act_dim, config, device="auto", seed=42):
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
                    c.get("max_grad_norm", 0.5), c.get("normalize_advantage", True), device, seed)
    elif name == "sac":
        return SAC(obs_dim, act_dim, h, act, c.get("lr_actor", 3e-4), c.get("lr_critic", 3e-4),
                    c.get("lr_alpha", 3e-4), gamma, c.get("tau", 0.005), c.get("batch_size", 256),
                    c.get("buffer_size", 1000000), c.get("learning_starts", 10000),
                    c.get("auto_entropy", True), c.get("init_alpha", 0.2), device, seed)
    elif name == "td3":
        return TD3(obs_dim, act_dim, h, act, c.get("lr_actor", 3e-4), c.get("lr_critic", 3e-4),
                    gamma, c.get("tau", 0.005), c.get("batch_size", 256),
                    c.get("buffer_size", 1000000), c.get("learning_starts", 10000),
                    c.get("policy_delay", 2), c.get("exploration_noise", 0.1),
                    c.get("target_noise", 0.2), c.get("noise_clip", 0.5), 1.0, device, seed)
    raise ValueError(f"지원하지 않는 알고리즘: {name}")


def train_ppo(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir):
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    for step in range(1, total_steps + 1):
        action, log_prob, value = algo.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        algo.buffer.add(obs, action, reward, value, log_prob, done)
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
            logger.dump(step)
        if step % eval_freq == 0:
            r = evaluator.evaluate(algo.select_action, step)
            for k, v in r.items():
                logger.log_scalar(k, v, step)
            print(f"  [평가] Return: {r['eval/mean_return']:.1f} ± {r['eval/std_return']:.1f}")
        if step % (eval_freq * 5) == 0:
            algo.total_steps = step
            algo.save(os.path.join(save_dir, "models", f"model_{step}.pt"))

    algo.total_steps = total_steps
    algo.save(os.path.join(save_dir, "models", "model_final.pt"))


def train_offpolicy(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir):
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

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

        if step >= algo.learning_starts:
            metrics = algo.update()
            for k, v in metrics.items():
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
            algo.save(os.path.join(save_dir, "models", f"model_{step}.pt"))

    algo.total_steps = total_steps
    algo.save(os.path.join(save_dir, "models", "model_final.pt"))


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
    exp_name = f"{args.algo}_{args.env}{rs}{ds}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"{'='*60}\n  알고리즘: {args.algo.upper()} | 환경: {args.env} | 시드: {args.seed}")
    print(f"  총 스텝: {total_steps:,} | 저장: {save_dir}\n{'='*60}")

    reward_fn = get_reward_fn(args.env, args.reward_type) if args.reward_type else None
    dr_cfg = config.get("domain_randomization", {}) if args.domain_rand else None

    env = make_env(args.env, args.seed, args.normalize_obs, gamma=cc.get("gamma", 0.99),
                   domain_randomization=args.domain_rand, dr_config=dr_cfg, custom_reward_fn=reward_fn)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    algo = create_algorithm(args.algo, obs_dim, act_dim, config, cc.get("device", "auto"), args.seed)
    if args.load_model:
        algo.load(args.load_model)

    logger = Logger(os.path.join(save_dir, "logs"), exp_name, args.tensorboard)
    evaluator = Evaluator(args.env, cc.get("eval_episodes", 10), args.seed + 1000)

    try:
        if args.algo == "ppo":
            train_ppo(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir)
        else:
            train_offpolicy(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir)
    finally:
        logger.close()
        evaluator.close()
        env.close()
    print("\n학습 완료!")


if __name__ == "__main__":
    main()
