"""
공통 신경망 모듈

모든 알고리즘에서 재사용할 수 있는 네트워크 빌딩 블록을 정의합니다.
- MLP: 범용 다층 퍼셉트론
- GaussianActor: 가우시안 분포 기반 확률적 정책 (PPO, SAC)
- DeterministicActor: 결정적 정책 (TD3)
- Critic (QNetwork): Q-value 네트워크
- VNetwork: Value 네트워크 (PPO)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple

LOG_STD_MIN = -4   # σ_min ≈ 0.018 (was -20 → σ=2e-9: log_prob→+326 → alpha 폭발 원인)
LOG_STD_MAX = 2


def get_activation(name: str) -> nn.Module:
    """활성화 함수를 이름으로 반환합니다."""
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"지원하지 않는 활성화 함수: {name}. 사용 가능: {list(activations.keys())}")
    return activations[name]


def init_weights(module: nn.Module, gain: float = 1.0):
    """Orthogonal 초기화 (PPO에서 주로 사용)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """범용 다층 퍼셉트론 (Multi-Layer Perceptron)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: nn.Module = None,
        apply_ortho_init: bool = False,
    ):
        super().__init__()
        activation_fn = get_activation(activation)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

        if apply_ortho_init:
            self._apply_ortho_init()

    def _apply_ortho_init(self):
        """히든 레이어에 Orthogonal 초기화, 출력 레이어에 gain=0.01 적용."""
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            init_weights(layer, gain=np.sqrt(2))
        init_weights(linear_layers[-1], gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """
    가우시안 분포 기반 확률적 정책 네트워크.

    PPO: log_std를 독립 파라미터로 관리 (state-independent)
    SAC: log_std를 네트워크 출력으로 생성 (state-dependent) + reparameterization trick
         max_action으로 tanh 출력을 환경 action space에 맞게 스케일링.
         (예: Humanoid-v5 = 0.4 → 출력 범위 [-0.4, 0.4])
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        state_dependent_std: bool = False,
        max_action: float = 1.0,
        apply_ortho_init: bool = False,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.max_action = max_action
        activation_fn = get_activation(activation)

        # 공유 히든 레이어
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Mean 헤드
        self.mean_head = nn.Linear(prev_dim, act_dim)

        if state_dependent_std:
            # SAC 스타일: log_std도 네트워크 출력
            self.log_std_head = nn.Linear(prev_dim, act_dim)
        else:
            # PPO 스타일: log_std는 학습 가능한 독립 파라미터
            # [FIX 3] Humanoid-v5 액션 스페이스(±0.4)에 맞게 초기 std를 줄임.
            # zeros(→ std=1.0)이면 초기 샘플이 [-3,3] 범위로 액션 경계를 크게 벗어나
            # MuJoCo ctrlrange 클리핑 후에도 최대 토크만 랜덤 방향으로 가해져 즉시 쓰러짐.
            # -1.0 → std=e^-1≈0.37: 액션 경계(±0.4)와 비슷한 스케일로 초반 안정성 확보.
            self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

        # PPO log_std 상한 클램프 상수 (state_dependent_std=False 전용)
        # std ≤ e^(-0.5) ≈ 0.61로 제한: MuJoCo 내부 클리핑과 조합 시 entropy 폭발 방지.
        # 경계값 액션이 log_std 그래디언트를 양수로 편향시키는 효과를 상한으로 차단.
        self._ppo_log_std_max = -0.5

        if apply_ortho_init:
            # PPO 권장: 히든 gain=sqrt(2), 출력 헤드 gain=0.01
            for layer in [m for m in self.shared if isinstance(m, nn.Linear)]:
                init_weights(layer, gain=np.sqrt(2))
            init_weights(self.mean_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """평균과 log 표준편차를 반환합니다."""
        features = self.shared(obs)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            # PPO: 상한을 _ppo_log_std_max(-0.5)로 고정해 entropy 폭발 방지.
            # MuJoCo 내부 클리핑(±0.4)과 상호작용 시 log_std가 단조 증가하는 현상을 차단.
            log_std = torch.clamp(self.log_std, LOG_STD_MIN, self._ppo_log_std_max).expand_as(mean)

        return mean, log_std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """가우시안 분포 객체를 반환합니다."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        행동을 샘플링하고 log 확률을 반환합니다.

        PPO용: 일반 샘플링
        """
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def rsample_with_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SAC용 reparameterization trick 샘플링.
        tanh squashing + max_action 스케일링 적용 후 보정된 log 확률 반환.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterization trick
        z = dist.rsample()
        tanh_z = torch.tanh(z)
        # max_action으로 스케일링 → 환경 action space와 일치
        action = self.max_action * tanh_z

        # tanh squashing에 의한 log 확률 보정
        # a = max_action * tanh(z) 변환의 야코비안: da/dz = max_action * (1 - tanh²(z))
        # log|da/dz| = log(max_action) + log(1 - tanh²(z))
        # [FIX 4] max_action ≠ 1.0일 때 log(max_action) 항을 반드시 포함해야 함.
        # 누락 시 log_prob가 act_dim * |log(max_action)| 만큼 편향(Humanoid: +15.57)되어
        # SAC alpha가 정책을 과도하게 결정론적(std≈0.003)으로 압축함 → 탐험 부족.
        log_prob = dist.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1) - z.shape[-1] * np.log(self.max_action)

        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PPO용: 주어진 행동에 대한 log 확률과 엔트로피를 반환합니다.
        """
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class DeterministicActor(nn.Module):
    """
    결정적 정책 네트워크 (TD3용).
    출력에 tanh를 적용하여 [-1, 1] 범위로 제한합니다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        max_action: float = 1.0,
    ):
        super().__init__()
        self.max_action = max_action
        self.net = MLP(obs_dim, act_dim, hidden_dims, activation, output_activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(obs)


class QNetwork(nn.Module):
    """
    Q-value 네트워크 (Critic).
    state-action 쌍을 입력받아 Q 값을 출력합니다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden_dims, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """
    Twin Q-Network (SAC, TD3용).
    두 개의 독립적인 Q-network로 과대추정을 방지합니다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [512, 512],
        activation: str = "relu",
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims, activation)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q1 네트워크만 사용 (정책 업데이트 시)."""
        return self.q1(obs, action)


class VNetwork(nn.Module):
    """Value 네트워크 (PPO용). 상태만 입력받아 V 값을 출력합니다."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        apply_ortho_init: bool = False,
    ):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden_dims, activation, apply_ortho_init=apply_ortho_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
