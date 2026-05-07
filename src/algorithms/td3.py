"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 구현

핵심 특징:
- Off-policy, Deterministic Policy Gradient
- Twin Q-Network으로 Q값 과대추정 방지
- Delayed Policy Update (매 d번째 critic 업데이트마다)
- Target Policy Smoothing (target action에 노이즈 추가)

Reference: Fujimoto et al., "Addressing Function Approximation Error in
           Actor-Critic Methods", 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
import copy

from .base import BaseAlgorithm
from ..common.networks import DeterministicActor, TwinQNetwork
from ..common.buffer import ReplayBuffer


class TD3(BaseAlgorithm):
    """Twin Delayed DDPG 알고리즘."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        max_action: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(obs_dim, act_dim, device, seed)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.max_action = max_action

        # --- 네트워크 초기화 ---

        # 결정적 정책 (Actor)
        self.actor = DeterministicActor(
            obs_dim, act_dim, hidden_dims, activation, max_action,
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        for param in self.actor_target.parameters():
            param.requires_grad = False

        # Twin Q-Network (Critic)
        self.critic = TwinQNetwork(
            obs_dim, act_dim, hidden_dims, activation,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # --- 옵티마이저 ---
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # --- Replay Buffer ---
        self.buffer = ReplayBuffer(
            obs_dim, act_dim, buffer_size,
            device=str(self.device),
        )

        # 업데이트 카운터 (policy delay 추적)
        self.update_count = 0

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """행동을 선택합니다."""
        with torch.no_grad():
            obs_tensor = self._to_tensor(obs).unsqueeze(0)
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()

        if not deterministic:
            # 가우시안 탐색 노이즈 추가
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self) -> Dict[str, float]:
        """
        Replay Buffer에서 미니배치를 샘플링하여 업데이트합니다.

        Returns:
            학습 메트릭
        """
        self.update_count += 1
        batch = self.buffer.sample(self.batch_size)

        # --- Critic 업데이트 (매 스텝) ---
        critic_loss = self._update_critic(batch)

        # --- Actor 업데이트 (매 d번째 스텝) ---
        actor_loss = 0.0
        if self.update_count % self.policy_delay == 0:
            actor_loss = self._update_actor(batch)

            # Target 네트워크 소프트 업데이트 (Actor 업데이트 시에만)
            self._soft_update()

        return {
            "loss/critic": critic_loss,
            "loss/actor": actor_loss,
        }

    def _update_critic(self, batch) -> float:
        """
        Twin Q-Network 업데이트.

        Target Policy Smoothing:
            a' = clip(π_target(s') + clip(ε, -c, c), -max, max)
            y = r + γ * min(Q1_target(s', a'), Q2_target(s', a'))
        """
        with torch.no_grad():
            # Target 행동 + smoothing noise
            next_action = self.actor_target(batch.next_observations)
            noise = torch.randn_like(next_action) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Target Q 값 (두 Q 중 최솟값)
            next_q1, next_q2 = self.critic_target(batch.next_observations, next_action)
            next_q = torch.min(next_q1, next_q2)

            # TD target
            target_q = batch.rewards + self.gamma * (1.0 - batch.dones) * next_q

        # 현재 Q 값
        q1, q2 = self.critic(batch.observations, batch.actions)

        # Twin Q 손실
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch) -> float:
        """
        결정적 정책 업데이트.

        max E[Q1(s, π(s))]  (Q1만 사용)
        """
        # Critic 파라미터 고정
        for param in self.critic.parameters():
            param.requires_grad = False

        action = self.actor(batch.observations)
        q1 = self.critic.q1_forward(batch.observations, action)
        actor_loss = -q1.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic 파라미터 다시 활성화
        for param in self.critic.parameters():
            param.requires_grad = True

        return actor_loss.item()

    def _soft_update(self):
        """Actor와 Critic Target 네트워크 소프트 업데이트."""
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def _get_save_state(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def _load_save_state(self, state: Dict[str, Any]):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_count = state.get("update_count", 0)
