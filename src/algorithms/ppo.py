"""
PPO (Proximal Policy Optimization) 구현

핵심 특징:
- On-policy Actor-Critic
- Clipped Surrogate Objective
- Generalized Advantage Estimation (GAE)
- 다중 에포크 미니배치 업데이트

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

from .base import BaseAlgorithm
from ..common.networks import GaussianActor, VNetwork
from ..common.buffer import RolloutBuffer


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization 알고리즘."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        entropy_coef: float = 0.0,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(obs_dim, act_dim, device, seed)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        # 네트워크 초기화
        self.actor = GaussianActor(
            obs_dim, act_dim, hidden_dims, activation,
            state_dependent_std=False,  # PPO: state-independent std
        ).to(self.device)

        self.critic = VNetwork(
            obs_dim, hidden_dims, activation,
        ).to(self.device)

        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 롤아웃 버퍼
        self.buffer = RolloutBuffer(
            obs_dim, act_dim, n_steps, gamma, gae_lambda,
            device=str(self.device),
        )

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """행동을 선택합니다."""
        with torch.no_grad():
            obs_tensor = self._to_tensor(obs).unsqueeze(0)

            if deterministic:
                mean, _ = self.actor(obs_tensor)
                action = mean.squeeze(0)
                return action.cpu().numpy()

            action, log_prob = self.actor.sample(obs_tensor)
            value = self.critic(obs_tensor)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0).cpu().item(),
            value.squeeze(0).cpu().item(),
        )

    def update(self, last_value: float, last_done: bool) -> Dict[str, float]:
        """
        수집된 롤아웃 데이터로 정책과 가치 함수를 업데이트합니다.

        Args:
            last_value: 마지막 관측값의 가치 추정
            last_done: 마지막 스텝의 종료 여부

        Returns:
            학습 메트릭
        """
        # GAE 계산
        self.buffer.compute_gae(last_value, last_done)

        # 다중 에포크 업데이트
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # 현재 정책에서의 log 확률과 엔트로피
                new_log_probs, entropy = self.actor.evaluate(
                    batch.observations, batch.actions
                )
                new_values = self.critic(batch.observations)

                # 어드밴티지 정규화
                advantages = batch.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- Policy Loss (Clipped Surrogate) ---
                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value Loss ---
                value_loss = F.mse_loss(new_values, batch.returns)

                # --- 전체 손실 ---
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                # 그래디언트 업데이트
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # 그래디언트 클리핑
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 버퍼 초기화
        self.buffer.reset()

        return {
            "loss/policy": total_policy_loss / n_updates,
            "loss/value": total_value_loss / n_updates,
            "loss/entropy": total_entropy / n_updates,
        }

    def _get_save_state(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def _load_save_state(self, state: Dict[str, Any]):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
