"""
SAC (Soft Actor-Critic) 구현

핵심 특징:
- Off-policy, Maximum Entropy RL
- Twin Q-Network으로 과대추정 방지
- Reparameterization trick으로 정책 그래디언트 계산
- 자동 엔트로피 계수 조절

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
           Deep Reinforcement Learning with a Stochastic Actor", 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
import copy

from .base import BaseAlgorithm
from ..common.networks import GaussianActor, TwinQNetwork
from ..common.buffer import ReplayBuffer


class SAC(BaseAlgorithm):
    """Soft Actor-Critic 알고리즘."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        auto_entropy: bool = True,
        init_alpha: float = 0.2,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(obs_dim, act_dim, device, seed)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.auto_entropy = auto_entropy

        # --- 네트워크 초기화 ---

        # 확률적 정책 (state-dependent std + reparameterization)
        self.actor = GaussianActor(
            obs_dim, act_dim, hidden_dims, activation,
            state_dependent_std=True,  # SAC: state-dependent std
        ).to(self.device)

        # Twin Q-Network
        self.critic = TwinQNetwork(
            obs_dim, act_dim, hidden_dims, activation,
        ).to(self.device)

        # Target Q-Network (소프트 업데이트 대상)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        # Target 네트워크는 그래디언트 계산 불필요
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # --- 옵티마이저 ---
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # --- 엔트로피 계수 (α) ---
        if auto_entropy:
            # 자동 조절: target entropy = -dim(A)
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32, device=self.device,
            )

        # --- Replay Buffer ---
        self.buffer = ReplayBuffer(
            obs_dim, act_dim, buffer_size,
            device=str(self.device),
        )

    @property
    def alpha(self) -> torch.Tensor:
        """현재 엔트로피 계수."""
        return self.log_alpha.exp()

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
                action = torch.tanh(mean)  # tanh squashing
            else:
                action, _ = self.actor.rsample_with_log_prob(obs_tensor)

        return action.squeeze(0).cpu().numpy()

    def update(self) -> Dict[str, float]:
        """
        Replay Buffer에서 미니배치를 샘플링하여 업데이트합니다.

        Returns:
            학습 메트릭
        """
        batch = self.buffer.sample(self.batch_size)

        # --- Critic 업데이트 ---
        critic_loss = self._update_critic(batch)

        # --- Actor 업데이트 ---
        actor_loss, log_prob_mean = self._update_actor(batch)

        # --- 엔트로피 계수 업데이트 ---
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = self._update_alpha(log_prob_mean)

        # --- Target 네트워크 소프트 업데이트 ---
        self._soft_update()

        return {
            "loss/critic": critic_loss,
            "loss/actor": actor_loss,
            "loss/alpha": alpha_loss,
            "info/alpha": self.alpha.item(),
            "info/log_prob": log_prob_mean,
        }

    def _update_critic(self, batch) -> float:
        """
        Twin Q-Network 업데이트.

        y = r + γ * (min(Q1_target, Q2_target) - α * log π(a'|s'))
        """
        with torch.no_grad():
            # 다음 행동 샘플링 (현재 정책에서)
            next_action, next_log_prob = self.actor.rsample_with_log_prob(
                batch.next_observations
            )
            # Target Q 값 계산
            next_q1, next_q2 = self.critic_target(batch.next_observations, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
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

    def _update_actor(self, batch) -> tuple:
        """
        정책 업데이트.

        max E[Q(s, π(s)) - α * log π(a|s)]
        """
        # Critic 파라미터 고정 (Actor 업데이트 시)
        for param in self.critic.parameters():
            param.requires_grad = False

        action, log_prob = self.actor.rsample_with_log_prob(batch.observations)
        q1 = self.critic.q1_forward(batch.observations, action)

        actor_loss = (self.alpha.detach() * log_prob - q1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic 파라미터 다시 활성화
        for param in self.critic.parameters():
            param.requires_grad = True

        return actor_loss.item(), log_prob.mean().item()

    def _update_alpha(self, log_prob_mean: float) -> float:
        """엔트로피 계수 자동 조절."""
        # log_prob_mean은 float이므로 텐서로 변환하여 그래디언트 계산
        log_prob_tensor = torch.tensor(log_prob_mean, device=self.device)
        alpha_loss = -(self.log_alpha * (log_prob_tensor + self.target_entropy))

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def _soft_update(self):
        """Target 네트워크 소프트 업데이트 (Polyak averaging)."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def _get_save_state(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.auto_entropy:
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def _load_save_state(self, state: Dict[str, Any]):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_(self.auto_entropy)
        if self.auto_entropy and "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
