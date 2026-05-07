"""
정책 평가 모듈

학습 중 주기적으로 정책을 평가하여 성능을 측정합니다.
Deterministic rollout으로 평가하여 탐색 노이즈의 영향을 제거합니다.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Callable
import torch


class Evaluator:
    """
    강화학습 정책 평가기.

    학습 중 주기적으로 별도의 평가 환경에서
    deterministic rollout을 수행하여 성능을 측정합니다.
    """

    def __init__(
        self,
        env_id: str,
        n_eval_episodes: int = 10,
        seed: int = 0,
    ):
        """
        Args:
            env_id: 평가 환경 ID
            n_eval_episodes: 평가 에피소드 수
            seed: 평가 환경 시드
        """
        self.env = gym.make(env_id)
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.eval_history: List[Dict] = []

    def evaluate(
        self,
        select_action_fn: Callable,
        step: int,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        정책을 평가합니다.

        Args:
            select_action_fn: 관측값 -> 행동 함수 (deterministic 모드)
            step: 현재 학습 스텝 (로깅용)
            deterministic: 결정적 행동 사용 여부

        Returns:
            평가 결과 딕셔너리
        """
        episode_returns = []
        episode_lengths = []

        for ep in range(self.n_eval_episodes):
            obs, _ = self.env.reset(seed=self.seed + ep)
            done = False
            episode_return = 0.0
            episode_length = 0

            while not done:
                action = select_action_fn(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
                done = terminated or truncated

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        results = {
            "eval/mean_return": float(np.mean(episode_returns)),
            "eval/std_return": float(np.std(episode_returns)),
            "eval/min_return": float(np.min(episode_returns)),
            "eval/max_return": float(np.max(episode_returns)),
            "eval/mean_length": float(np.mean(episode_lengths)),
            "eval/step": step,
        }

        self.eval_history.append(results)
        return results

    def get_best_return(self) -> float:
        """지금까지의 최고 평균 리턴을 반환합니다."""
        if not self.eval_history:
            return float("-inf")
        return max(h["eval/mean_return"] for h in self.eval_history)

    def close(self):
        """환경을 정리합니다."""
        self.env.close()
