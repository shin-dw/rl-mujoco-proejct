"""
커스텀 리워드 함수 모듈

Humanoid-v5 환경에 특화된 커스텀 리워드를 정의합니다.
기본 reward와의 비교를 위해 CustomRewardWrapper와 연동됩니다.
"""

import numpy as np
from typing import Dict, Any


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
    - y축 직진 유도
    - 좌우 고관절 대칭성
    """
    forward_reward = reward

    # PPO unbounded action → ctrlrange(±0.4)로 clip 후 사용
    clipped = np.clip(action, -0.4, 0.4)

    # z축 높이 유지 보상
    uprightness_bonus = 0.0
    if len(next_obs) > 2:
        height = next_obs[0] if len(next_obs) > 0 else 1.2
        target_height = 1.2
        uprightness_bonus = -0.5 * (height - target_height) ** 2

    # 에너지 절약 페널티 (clipped action 사용 — base의 control_cost와 일관성 유지)
    action_penalty = -0.02 * np.sum(clipped ** 2)

    # y축 이탈 페널티 (x방향 직진 유도)
    y_velocity_penalty = -0.3 * abs(info.get("y_velocity", 0.0))

    # 좌우 고관절 대칭성 페널티 (clipped action 사용)
    symmetry_penalty = -0.1 * abs(clipped[3] - clipped[7])

    return forward_reward + uprightness_bonus + action_penalty + y_velocity_penalty + symmetry_penalty


class HumanoidStableGait:
    """
    안정적 보행 패턴 리워드 (Humanoid).

    이전 action을 기억하여 행동 변화량 페널티를 계산합니다.
    - 연속된 action 간 변화량이 클수록 페널티 (진짜 smoothness)
    """

    def __init__(self):
        self.prev_action = None

    def __call__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        forward_reward = reward

        # PPO unbounded action → ctrlrange(±0.4)로 clip 후 사용
        clipped = np.clip(action, -0.4, 0.4)

        # action 크기 페널티 (clipped 사용)
        smoothness_penalty = -0.01 * np.sum(clipped ** 2)

        # 이전 action 대비 변화량 페널티 (clipped 사용)
        if self.prev_action is not None:
            action_diff_penalty = -0.1 * np.sum((clipped - self.prev_action) ** 2)
        else:
            action_diff_penalty = 0.0

        self.prev_action = clipped.copy()

        return forward_reward + smoothness_penalty + action_diff_penalty


def humanoid_run_forward(
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    info: Dict[str, Any],
) -> float:
    """
    달리기 특화 보상 (좀비/펭귄 걸음 완벽 치료용)
    """
    x_vel = info.get("x_velocity", 0.0)
    y_vel = info.get("y_velocity", 0.0)
    
    # 기본 생존 보상(+5.0)이 포함된 base
    base = reward 
    
    # 1. 전진 보상 강화 (직진 의지 펌핑)
    velocity_bonus = 3.0 * max(x_vel, 0.0)
    
    # 2. Y축 이탈 맹독성 페널티 (대각선/게걸음 완벽 차단)
    # 기존 -0.3에서 -1.0으로 대폭 올려 옆으로 새는 즉시 엄청난 감점을 줍니다.
    y_velocity_penalty = -1.0 * abs(y_vel)
    
    # 3. ★ 셔플링(좀비걸음) 사형 선고 ★
    # 속도가 0.5m/s 이하로 꼼지락거리면 생존 보상을 깎아버립니다. 
    # 이제 에이전트는 살기 위해서 무조건 무릎을 굽히고 속도를 내야만 합니다.
    shuffle_penalty = 0.0
    if x_vel < 0.5:
        shuffle_penalty = -2.0 
        
    # 액션 클리핑 (리워드 해킹 방지)
    clipped = np.clip(action, -0.4, 0.4)
    
    # 4. 고관절 교차(Pitch: 5, 9) 유도
    hip_alternation_bonus = -1.0 * (clipped[5] * clipped[9])
    
    # 5. 무릎 관절 사용 강제 넛지 (Pitch: 6, 10)
    # 6번(오른무릎)과 10번(왼무릎) 관절에 힘을 줄 때마다 소소한 보너스를 줍니다.
    # 에이전트가 "어? 무릎을 굽히니까 점수를 주네?" 하고 깨닫게 만듭니다.
    knee_usage_bonus = 0.2 * (abs(clipped[6]) + abs(clipped[10]))
    
    return base + velocity_bonus + y_velocity_penalty + shuffle_penalty + hip_alternation_bonus + knee_usage_bonus


# =============================================================================
# 리워드 함수 레지스트리
# =============================================================================

REWARD_REGISTRY = {
    "Humanoid-v5": {
        "balanced_walk": humanoid_balanced_walk,
        "stable_gait": HumanoidStableGait,
        "run_forward": humanoid_run_forward,
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

    fn = env_rewards[reward_type]
    # 클래스인 경우 인스턴스 생성 (prev_action 등 상태 유지용)
    if isinstance(fn, type):
        return fn()
    return fn
