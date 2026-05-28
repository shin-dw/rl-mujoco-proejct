"""
정책 평가 모듈

학습 중 주기적으로 정책을 평가하여 성능을 측정합니다.
Deterministic rollout으로 평가하여 탐색 노이즈의 영향을 제거합니다.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Callable


def _find_normalize_wrapper(env):
    """래퍼 체인에서 NormalizeObservation 인스턴스를 탐색."""
    from .env_wrapper import NormalizeObservation
    current = env
    while current is not None:
        if isinstance(current, NormalizeObservation):
            return current
        current = getattr(current, "env", None)
    return None


class Evaluator:
    """
    강화학습 정책 평가기.

    학습 환경에 NormalizeObservation이 적용된 경우, 평가 환경에도
    동일한 래퍼를 적용하고 매 평가 직전에 running 통계를 동기화합니다.
    """

    def __init__(
        self,
        env_id: str,
        n_eval_episodes: int = 10,
        seed: int = 0,
        train_env=None,
    ):
        from .env_wrapper import NormalizeObservation

        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.eval_history: List[Dict] = []

        raw_env = gym.make(env_id)

        # 학습 환경에 정규화 래퍼가 있으면 평가 환경에도 동일하게 적용
        self._train_norm = _find_normalize_wrapper(train_env) if train_env is not None else None
        if self._train_norm is not None:
            self.env = NormalizeObservation(raw_env)
            self._eval_norm = _find_normalize_wrapper(self.env)
        else:
            self.env = raw_env
            self._eval_norm = None

    def _sync_obs_stats(self):
        """학습 환경의 running 통계를 평가 환경에 복사."""
        if self._train_norm is None or self._eval_norm is None:
            return
        self._eval_norm.running_mean = self._train_norm.running_mean.copy()
        self._eval_norm.running_var = self._train_norm.running_var.copy()
        self._eval_norm.count = self._train_norm.count

    def evaluate(
        self,
        select_action_fn: Callable,
        step: int,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        self._sync_obs_stats()

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
        if not self.eval_history:
            return float("-inf")
        return max(h["eval/mean_return"] for h in self.eval_history)

    def close(self):
        self.env.close()
