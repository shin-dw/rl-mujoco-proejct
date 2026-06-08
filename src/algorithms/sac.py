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
        target_entropy: float = None,
        max_action: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        gradient_steps: int = 1,
    ):
        super().__init__(obs_dim, act_dim, device, seed)
        self.max_action = max_action

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.auto_entropy = auto_entropy
        self.gradient_steps = gradient_steps  # update() 내부에서 반복할 gradient step 수

        # ⭐⭐⭐ [수정된 부분 1] Humanoid 환경에 맞게 보상 스케일 축소 파라미터 추가
        self.reward_scale = 0.1

        # --- 네트워크 초기화 ---

        # 확률적 정책 (state-dependent std + reparameterization)
        self.actor = GaussianActor(
            obs_dim, act_dim, hidden_dims, activation,
            state_dependent_std=True,  # SAC: state-dependent std
            max_action=max_action,     # 환경 action space에 맞게 스케일링
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
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
            else:
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
                action = self.max_action * torch.tanh(mean)  # tanh + max_action 스케일링
            else:
                action, _ = self.actor.rsample_with_log_prob(obs_tensor)

        return action.squeeze(0).cpu().numpy()

    def update(self) -> Dict[str, float]:
        """
        Replay Buffer에서 미니배치를 샘플링하여 업데이트합니다.

        gradient_steps 회 반복 후 .item()을 마지막에 한 번만 호출하여
        GPU-CPU 동기화 횟수를 최소화합니다.

        Returns:
            학습 메트릭
        """
        critic_loss_sum: torch.Tensor | None = None
        actor_loss_sum: torch.Tensor | None  = None
        alpha_loss_sum: torch.Tensor | None  = None
        last_log_prob:  torch.Tensor | None  = None

        for _ in range(self.gradient_steps):
            batch = self.buffer.sample(self.batch_size)

            # --- Critic 업데이트 ---
            cl = self._update_critic(batch)
            critic_loss_sum = cl if critic_loss_sum is None else critic_loss_sum + cl

            # --- Actor 업데이트 ---
            al, lp = self._update_actor(batch)
            actor_loss_sum = al if actor_loss_sum is None else actor_loss_sum + al
            last_log_prob = lp

            # --- 엔트로피 계수 업데이트 ---
            if self.auto_entropy:
                apl = self._update_alpha(lp)
                alpha_loss_sum = apl if alpha_loss_sum is None else alpha_loss_sum + apl

            # --- Target 네트워크 소프트 업데이트 ---
            self._soft_update()

        gs = self.gradient_steps
        # .item() 호출은 여기서 한 번만 (GPU-CPU 동기화 최소화)
        return {
            "loss/critic":    (critic_loss_sum / gs).item(),
            "loss/actor":     (actor_loss_sum  / gs).item(),
            "loss/alpha":     (alpha_loss_sum  / gs).item() if alpha_loss_sum is not None else 0.0,
            "info/alpha":     self.alpha.item(),
            "info/log_prob":  last_log_prob.item(),
        }

    def _update_critic(self, batch) -> torch.Tensor:
        """
        Twin Q-Network 업데이트.

        y = r + γ * (min(Q1_target, Q2_target) - α * log π(a'|s'))
        텐서를 반환 (.item()은 update()에서 한 번만 호출)
        """
        with torch.no_grad():
            # 다음 행동 샘플링 (현재 정책에서)
            next_action, next_log_prob = self.actor.rsample_with_log_prob(
                batch.next_observations
            )

            # ⭐⭐⭐ [수정된 부분 2] Log Prob이 음의 무한대로 발산하여 Q값이 폭발하는 것 방지
            next_log_prob = torch.clamp(next_log_prob, min=-20.0, max=2.0)

            # Target Q 값 계산
            next_q1, next_q2 = self.critic_target(batch.next_observations, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob

            # ⭐⭐⭐ [수정된 부분 3] 보상에 스케일을 곱하여 거대한 보상 단위 축소
            scaled_rewards = batch.rewards * self.reward_scale
            target_q = scaled_rewards + self.gamma * (1.0 - batch.dones) * next_q

        # 현재 Q 값
        q1, q2 = self.critic(batch.observations, batch.actions)

        # Twin Q 손실 — Huber loss 대신 MSE 사용
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # ⭐⭐⭐ [수정된 부분 4] Critic Gradient Clipping을 40.0에서 1.0으로 복구하여 급격한 파라미터 붕괴 방지
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        return critic_loss.detach()  # .item() 호출 없이 텐서 반환

    def _update_actor(self, batch) -> tuple:
        """
        정책 업데이트.

        max E[min(Q1,Q2)(s, π(s)) - α * log π(a|s)]
        SAC 논문(Haarnoja et al. 2018)에 따라 min(Q1,Q2)를 사용하여 과대추정 방지.
        Actor 업데이트 시 critic parameter에 gradient가 흐르지 않도록 freeze.
        텐서 튜플을 반환 (.item()은 update()에서 한 번만 호출)
        """
        action, log_prob = self.actor.rsample_with_log_prob(batch.observations)

        # ⭐⭐⭐ [수정된 부분 5] Actor 업데이트 시에도 Log Prob 발산 방지 적용
        log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)

        # critic freeze: actor loss backward 시 critic에 불필요한 gradient 방지
        for param in self.critic.parameters():
            param.requires_grad_(False)

        q1, q2 = self.critic(batch.observations, action)
        q_min = torch.min(q1, q2)  # SAC: min(Q1,Q2)로 과대추정 방지
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(True)

        return actor_loss.detach(), log_prob.mean().detach()  # 텐서 반환

    def _update_alpha(self, log_prob: torch.Tensor) -> torch.Tensor:
        """엔트로피 계수 자동 조절.

        log_prob: _update_actor()가 반환한 텐서 (float 변환 없이 직접 사용)
        텐서를 반환 (.item()은 update()에서 한 번만 호출)
        """
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach())

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 이미 넓게 풀어둔 부분 유지
        self.log_alpha.data.clamp_(min=-20.0, max=5.0)

        return alpha_loss.detach()  # 텐서 반환

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