"""
공통 신경망 모듈

모든 알고리즘에서 재사용할 수 있는 네트워크 빌딩 블록을 정의합니다.
- MLP: 범용 다층 퍼셉트론
- GaussianActor: 가우시안 분포 기반 확률적 정책 (PPO, SAC)
- DeterministicActor: 결정적 정책 (TD3)
- Critic (QNetwork): Q-value 네트워크
- VNetwork: Value 네트워크 (PPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple

LOG_STD_MIN = -20
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """
    가우시안 분포 기반 확률적 정책 네트워크.

    PPO: log_std를 독립 파라미터로 관리 (state-independent)
    SAC: log_std를 네트워크 출력으로 생성 (state-dependent) + reparameterization trick
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        state_dependent_std: bool = False,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
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
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """평균과 log 표준편차를 반환합니다."""
        features = self.shared(obs)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            log_std = self.log_std.expand_as(mean)

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
        tanh squashing 적용 후 보정된 log 확률 반환.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterization trick
        z = dist.rsample()
        action = torch.tanh(z)

        # tanh squashing에 의한 log 확률 보정
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

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
        hidden_dims: List[int] = [256, 256],
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
    ):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
