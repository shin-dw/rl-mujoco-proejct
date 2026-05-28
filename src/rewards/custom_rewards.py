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
    달리기 특화 보상 (Humanoid-v5).

    문제: 기본 healthy_reward(5.0/step)가 전진 보상을 압도해
         에이전트가 발을 조금씩만 움직이는 셔플링에 수렴함.

    해결 전략:
    - 전진 속도에 가중치(2.0×) → 달리기가 셔플링보다 유리하되 과도한 공격성 억제
    - x_vel < 0.5m/s 구간에 완만한 정체 페널티(-1.0) → 셔플링 수익성 제거 (기존 -3.0은 과도)
    - y축 이탈 억제 → 직진 달리기 유도
    - 좌우 고관절 교차 보상 → 호핑 억제, 자연스러운 교차 보행 유도
    - action_penalty 강화(0.1×) → MuJoCo 실제 control_cost(0.1)에 맞춤, 큰 토크 억제

    설계 원칙 (v4 — 안전한 교차보행 유도):
    - velocity_bonus가 주 신호 (달리기 동기 부여)
    - hip_alternation은 작은 넛지 수준 (velocity의 ~8%) → reward hacking 방지
    - 페널티 없음 — 초기 학습을 방해하지 않음
    - base에 이미 control_cost(-0.1×Σa²) 포함 → action_penalty 추가 금지

    계수 설계 근거:
      velocity_bonus (x_vel=1.0): 2.0/step         ← 주 신호
      hip_alternation max:         ±0.16/step       ← 부 신호 (8%)
      → hip이 달리기를 지배하지 않으므로 reward hacking 불가

    보상 비교 (per step):
      셔플링    x_vel=0.3: base(5.3) + vel(0.6) + hip(~0)   ≈  5.9
      보행      x_vel=1.0: base(6.0) + vel(2.0) + hip(+0.16) ≈  8.2
      교차달리기 x_vel=2.5: base(7.5) + vel(5.0) + hip(+0.16) ≈ 12.7
    """
    x_vel = info.get("x_velocity", 0.0)

    # 기본 MuJoCo 보상 (healthy_reward + forward + control_cost 이미 포함)
    base = reward

    # 전진 속도 보너스 (주 신호)
    velocity_bonus = 2.0 * max(x_vel, 0.0)

    # ★ PPO 전용 주의사항:
    #   PPO actor는 unbounded Gaussian → action이 ±수십까지 가능
    #   MuJoCo는 ctrlrange(±0.4)로 clip하지만 reward_fn은 clip 전 raw action을 받음
    #   → action 값을 그대로 곱하면 reward가 수만으로 폭발 (reward hacking)
    #   → 반드시 MuJoCo ctrlrange로 clip 후 사용해야 함
    clipped = np.clip(action, -0.4, 0.4)

    # 좌우 고관절 교차 보상 (작은 넛지 — velocity의 ~8% 수준)
    # 반대 방향 → 곱 음수 → 보너스 양수 (교차 O)
    # 같은 방향 → 곱 양수 → 페널티 (호핑 억제)
    # max = 1.0 × 0.4² = ±0.16/step (bounded, hacking 불가)
    hip_alternation_bonus = -1.0 * (clipped[3] * clipped[7])

    return base + velocity_bonus + hip_alternation_bonus


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
