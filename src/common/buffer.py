"""
경험 버퍼 모듈

Off-policy 알고리즘(SAC, TD3)용 ReplayBuffer와
On-policy 알고리즘(PPO)용 RolloutBuffer를 구현합니다.
"""

import numpy as np
import torch
from typing import Dict, Optional, NamedTuple


class ReplayBufferSample(NamedTuple):
    """Replay Buffer에서 샘플링한 배치."""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class RolloutBufferSample(NamedTuple):
    """Rollout Buffer에서 샘플링한 배치."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBuffer:
    """
    Off-policy 알고리즘용 경험 재생 버퍼.

    고정 크기의 순환 버퍼로 (s, a, r, s', done) 전이를 저장하고,
    균일 랜덤 샘플링을 지원합니다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        buffer_size: int = 1_000_000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size
        self.device = device

        # NumPy 배열로 저장 (메모리 효율)
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0        # 현재 삽입 위치
        self.size = 0       # 현재 저장된 전이 수

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """전이 하나를 버퍼에 추가합니다."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> ReplayBufferSample:
        """균일 랜덤 샘플링으로 미니배치를 반환합니다."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return ReplayBufferSample(
            observations=torch.FloatTensor(self.observations[indices]).to(self.device),
            actions=torch.FloatTensor(self.actions[indices]).to(self.device),
            rewards=torch.FloatTensor(self.rewards[indices]).to(self.device),
            next_observations=torch.FloatTensor(self.next_observations[indices]).to(self.device),
            dones=torch.FloatTensor(self.dones[indices]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """
    On-policy 알고리즘(PPO)용 롤아웃 버퍼.

    한 롤아웃(n_steps)의 데이터를 수집하고,
    GAE(Generalized Advantage Estimation)를 계산한 뒤
    미니배치 단위로 반환합니다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.observations = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)

        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """한 스텝의 데이터를 추가합니다."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool):
        """
        GAE (Generalized Advantage Estimation)를 계산합니다.

        A_t = Σ (γλ)^l * δ_{t+l}
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        """
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_done = float(last_done)
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            # TD 오차
            delta = self.rewards[t] + self.gamma * next_value * (1.0 - next_done) - self.values[t]
            # GAE 재귀
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_gae
            self.advantages[t] = last_gae

        # 리턴 = 어드밴티지 + 밸류
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """데이터를 미니배치로 나누어 반환하는 제너레이터."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield RolloutBufferSample(
                observations=torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                actions=torch.FloatTensor(self.actions[batch_indices]).to(self.device),
                old_log_probs=torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                advantages=torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                returns=torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            )

    def reset(self):
        """버퍼를 초기화합니다."""
        self.ptr = 0
        self.full = False
