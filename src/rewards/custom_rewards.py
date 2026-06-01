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
    
    x_vel = info.get("x_velocity", 0.0)
    y_vel = info.get("y_velocity", 0.0)
    base = reward

    clipped = np.clip(action, -0.4, 0.4)

    # 1. 전진 속도 보너스 (그대로 유지)
    velocity_bonus = 3.0 * max(x_vel, 0.0)

    # 2. [수정] 후방 기울임(Leaning Backward) 철퇴 ────────────────────────
    # obs[1]~[4]는 몸통의 쿼터니언 (w, x, y, z)
    w = float(next_obs[1])
    x = float(next_obs[2])
    y = float(next_obs[3])
    z = float(next_obs[4])

    # 몸통의 로컬 Up(정수리 방향) 벡터가 전진 방향(X축)으로 얼마나 기울었는지 계산
    # ux > 0: 앞으로 숙임 (육상 선수 폼)
    # ux < 0: 뒤로 누움 (매트릭스 회피 폼)
    ux = 2.0 * (x * z + w * y)

    posture_penalty = 0.0
    if ux < 0:
        # 뒤로 눕는 자세에 대해 각도에 비례한 치명적인 페널티 부여
        posture_penalty = -10.0 * abs(ux)
    
    # 너무 극단적으로 앞으로 고꾸라지는 것만 방지 (약 60도 이상)
    torso_upright = 1.0 - 2.0 * (x ** 2 + y ** 2)
    if torso_upright < 0.5:
        posture_penalty -= 2.0

    # ── 3. 고관절 교차 보너스 및 캥거루 점프 사형 선고 ──
    hip_product = clipped[5] * clipped[9]
    
    if hip_product > 0:
        # 양 다리가 같은 방향으로 힘을 받음 = 양발 점프 중 (캥거루)
        # 속도 보너스를 다 깎아먹을 만큼 엄청난 치명적 감점을 부여합니다.
        hip_alternation_bonus = -20.0 * hip_product
    else:
        # 양 다리가 반대 방향으로 힘을 받음 = 정상 교차 중
        # 정상적인 교차 보행에는 기존과 비슷한 수준의 보너스를 줍니다.
        hip_alternation_bonus = -2.0 * hip_product

    # 4. 무릎 및 팔 스윙 (보조 역할로 유지)
    knee_usage_bonus = 0.5 * (abs(clipped[6]) + abs(clipped[10]))
    if len(clipped) > 16:
        arm_swing_bonus = -0.5 * (clipped[15] * clipped[5] + clipped[16] * clipped[9])
    else:
        arm_swing_bonus = 0.0

    # 5. Y축 및 셔플링 페널티 (유지)
    y_penalty = -1.0 * abs(y_vel)
    shuffle_penalty = -2.0 if x_vel < 0.5 else 0.0

    return (base + velocity_bonus + posture_penalty
            + hip_alternation_bonus + knee_usage_bonus
            + arm_swing_bonus + y_penalty + shuffle_penalty)


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