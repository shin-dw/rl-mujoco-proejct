"""
커스텀 리워드 함수 모듈

각 MuJoCo 환경에 특화된 커스텀 리워드를 정의합니다.
기본 reward와의 비교를 위해 CustomRewardWrapper와 연동됩니다.
"""

import numpy as np
from typing import Dict, Any


# =============================================================================
# HalfCheetah 커스텀 리워드
# =============================================================================

def halfcheetah_energy_efficient(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    에너지 효율 중심 리워드.

    기본 리워드에 에너지 효율 보너스를 추가합니다.
    - 행동의 L2 노름이 작을수록 보너스 증가
    - 전진 속도 대비 에너지 효율을 극대화
    """
    # 기본 리워드 (전진 속도 - 제어 비용)
    forward_reward = reward

    # 에너지 효율 보너스: 적은 에너지로 빠르게 달리기
    action_cost = np.sum(action ** 2)
    energy_bonus = -0.05 * action_cost  # 추가 에너지 페널티

    return forward_reward + energy_bonus


def halfcheetah_stability(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    자세 안정성 중심 리워드.

    기본 리워드에 자세 안정성 보너스를 추가합니다.
    - 몸체의 높이(z)가 안정적으로 유지되면 보너스
    - 급격한 회전을 방지하는 페널티
    """
    forward_reward = reward

    # 높이 안정성 보너스 (z 좌표가 목표 높이에 가까울수록)
    # HalfCheetah의 rootz는 obs[0]에 해당
    target_height = 0.0  # 기본 높이
    height_penalty = -0.1 * (next_obs[0] - target_height) ** 2

    # 회전 각속도 페널티 (급격한 회전 방지)
    angular_vel_penalty = -0.01 * np.sum(next_obs[8:14] ** 2)

    return forward_reward + height_penalty + angular_vel_penalty


# =============================================================================
# Ant 커스텀 리워드
# =============================================================================

def ant_directional(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    직진성 보상 리워드.

    기본 리워드에 직진성 보너스를 추가합니다.
    - x축 방향으로 직진할수록 보너스
    - y축 이탈 페널티
    """
    forward_reward = reward

    # y축 이탈 페널티 (직진 유도)
    y_velocity_penalty = -0.5 * abs(info.get("y_velocity", 0.0))

    # 높이 유지 보상 (넘어지지 않도록)
    height = info.get("z_distance_from_origin", next_obs[0] if len(next_obs) > 0 else 0.5)
    height_bonus = 0.0
    if 0.3 < height < 1.0:
        height_bonus = 0.1  # 적절한 높이 유지 시 보너스

    return forward_reward + y_velocity_penalty + height_bonus


def ant_energy_efficient(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    에너지 효율 중심 리워드 (Ant).

    다리 움직임의 에너지 효율을 극대화합니다.
    """
    forward_reward = reward

    # 관절 토크 페널티 강화
    action_cost = -0.1 * np.sum(action ** 2)

    return forward_reward + action_cost


# =============================================================================
# Humanoid 커스텀 리워드
# =============================================================================

def humanoid_balanced_walk(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    균형 잡힌 보행 리워드 (Humanoid).

    - 전진 속도 보상
    - 상체 안정성 보너스
    - 에너지 효율 보너스
    """
    forward_reward = reward

    # 상체 기울기 페널티 (직립 유지)
    # Humanoid의 상체 orientation은 quat으로 표현됨
    uprightness_bonus = 0.0
    if len(next_obs) > 2:
        # z축 높이 유지 보상
        height = next_obs[0] if len(next_obs) > 0 else 1.2
        target_height = 1.2
        uprightness_bonus = -0.5 * (height - target_height) ** 2

    # 에너지 절약 보너스
    action_penalty = -0.02 * np.sum(action ** 2)

    return forward_reward + uprightness_bonus + action_penalty


def humanoid_stable_gait(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    안정적 보행 패턴 리워드 (Humanoid).

    급격한 행동 변화를 억제하여 자연스러운 보행을 유도합니다.
    """
    forward_reward = reward

    # 행동 부드러움 보너스 (큰 행동 변화 페널티)
    smoothness_penalty = -0.01 * np.sum(action ** 2)

    return forward_reward + smoothness_penalty


# =============================================================================
# 리워드 함수 레지스트리
# =============================================================================

REWARD_REGISTRY = {
    "HalfCheetah-v5": {
        "energy_efficient": halfcheetah_energy_efficient,
        "stability": halfcheetah_stability,
    },
    "Ant-v5": {
        "directional": ant_directional,
        "energy_efficient": ant_energy_efficient,
    },
    "Humanoid-v5": {
        "balanced_walk": humanoid_balanced_walk,
        "stable_gait": humanoid_stable_gait,
    },
}


def get_reward_fn(env_id: str, reward_type: str):
    """환경과 리워드 타입으로 리워드 함수를 반환합니다."""
    if env_id not in REWARD_REGISTRY:
        raise ValueError(f"'{env_id}'에 대한 커스텀 리워드가 없습니다. 사용 가능: {list(REWARD_REGISTRY.keys())}")

    env_rewards = REWARD_REGISTRY[env_id]
    if reward_type not in env_rewards:
        raise ValueError(
            f"'{env_id}'에서 '{reward_type}' 리워드를 찾을 수 없습니다. "
            f"사용 가능: {list(env_rewards.keys())}"
        )

    return env_rewards[reward_type]
