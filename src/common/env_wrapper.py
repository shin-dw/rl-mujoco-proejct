"""
환경 래퍼 모듈

Gymnasium 환경을 감싸는 래퍼들을 정의합니다.
- NormalizeObservation: 관측값 정규화 (running mean/std)
- NormalizeReward: 보상 정규화
- DomainRandomizationWrapper: MuJoCo 물리 파라미터 무작위화
- CustomRewardWrapper: 커스텀 리워드 함수 적용
- make_env(): 환경 생성 팩토리 함수
"""

import gymnasium as gym
import numpy as np
from typing import Callable, Dict, Optional, Any


class NormalizeObservation(gym.Wrapper):
    """
    관측값을 Running Mean/Std로 정규화합니다.

    학습 중 관측값의 통계를 온라인으로 업데이트하며,
    (obs - mean) / std 형태로 정규화합니다.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(obs_shape, dtype=np.float64)
        self.running_var = np.ones(obs_shape, dtype=np.float64)
        self.count = 0

    def _update_stats(self, obs: np.ndarray):
        """Welford's online algorithm으로 통계를 업데이트합니다."""
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_stats(obs)
        return self._normalize(obs).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_stats(obs)
        return self._normalize(obs).astype(np.float32), reward, terminated, truncated, info


class NormalizeReward(gym.Wrapper):
    """
    보상을 Running Std로 정규화합니다.

    리턴(discounted return)의 분산을 추적하여
    보상을 표준편차로 나누어 정규화합니다.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_var = 1.0
        self.running_mean = 0.0
        self.count = 0
        self.discounted_return = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 할인된 리턴 추적
        self.discounted_return = reward + self.gamma * self.discounted_return
        self.count += 1
        delta = self.discounted_return - self.running_mean
        self.running_mean += delta / self.count
        delta2 = self.discounted_return - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count

        # 에피소드 종료 시 리턴 초기화
        if terminated or truncated:
            self.discounted_return = 0.0

        # running_var가 수렴하기 전(초기 수십 에피소드)에 0에 가까워지는 것을 방지.
        # max(running_var, 1.0) 로 클램핑하면 초기엔 정규화 없이 raw reward를 그대로 사용하고,
        # 분산이 충분히 추정된 이후 (>> 1.0) 부터 정규화가 효과를 발휘한다.
        std = np.sqrt(max(self.running_var, 1.0)) + self.epsilon
        normalized_reward = reward / std
        return obs, normalized_reward, terminated, truncated, info


class DomainRandomizationWrapper(gym.Wrapper):
    """
    MuJoCo 환경의 물리 파라미터를 매 에피소드마다 무작위화합니다.

    - 마찰 계수 (friction)
    - 바디 질량 (mass)
    - 관절 감쇠 (damping)
    """

    def __init__(
        self,
        env: gym.Env,
        friction_range: tuple = (0.7, 1.3),
        mass_range: tuple = (0.8, 1.2),
        damping_range: tuple = (0.75, 1.25),
    ):
        super().__init__(env)
        self.friction_range = friction_range
        self.mass_range = mass_range
        self.damping_range = damping_range

        # 기본 물리 파라미터를 저장
        model = self.unwrapped.model
        self.default_friction = model.geom_friction.copy()
        self.default_mass = model.body_mass.copy()
        self.default_damping = model.dof_damping.copy()

    def _randomize(self):
        """물리 파라미터를 무작위로 변경합니다."""
        model = self.unwrapped.model

        # 마찰 계수 무작위화
        friction_scale = np.random.uniform(*self.friction_range)
        model.geom_friction[:] = self.default_friction * friction_scale

        # 바디 질량 무작위화
        mass_scale = np.random.uniform(*self.mass_range)
        model.body_mass[:] = self.default_mass * mass_scale

        # 감쇠 무작위화
        damping_scale = np.random.uniform(*self.damping_range)
        model.dof_damping[:] = self.default_damping * damping_scale

    def reset(self, **kwargs):
        self._randomize()
        return self.env.reset(**kwargs)


class ScaleReward(gym.Wrapper):
    """
    보상을 고정 상수로 나누어 스케일링합니다.

    NormalizeReward와 달리 스케일이 학습 중 변하지 않으므로
    Replay Buffer와 함께 안전하게 사용할 수 있습니다.
    reward_scale=5.0이면 reward / 5.0 을 반환합니다.
    """

    def __init__(self, env: gym.Env, scale: float = 5.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        return obs, reward / self.scale, terminated, truncated, info


class CustomRewardWrapper(gym.Wrapper):
    """
    커스텀 리워드 함수를 적용하는 래퍼.

    reward_fn: (obs, action, reward, next_obs, info) -> new_reward
    """

    def __init__(self, env: gym.Env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._current_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # 커스텀 리워드 계산
        custom_reward = self.reward_fn(
            self._current_obs, action, reward, next_obs, info
        )
        info["original_reward"] = reward
        self._current_obs = next_obs
        return next_obs, custom_reward, terminated, truncated, info


def make_env(
    env_id: str,
    seed: int = 42,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    gamma: float = 0.99,
    domain_randomization: bool = False,
    dr_config: Optional[Dict] = None,
    custom_reward_fn: Optional[Callable] = None,
    reward_scale: float = 1.0,
) -> gym.Env:
    """
    환경 생성 팩토리 함수.

    다양한 래퍼를 조합하여 환경을 구성합니다.

    Args:
        env_id: Gymnasium 환경 ID (예: "HalfCheetah-v5")
        seed: 랜덤 시드
        normalize_obs: 관측값 정규화 여부
        normalize_reward: 보상 정규화 여부 (off-policy 알고리즘에는 비권장 — replay buffer와 비호환)
        gamma: 할인 계수 (보상 정규화에 사용)
        domain_randomization: Domain Randomization 적용 여부
        dr_config: Domain Randomization 설정
        custom_reward_fn: 커스텀 리워드 함수
        reward_scale: 고정 보상 스케일 (reward / scale). off-policy 알고리즘 Q값 폭발 방지용.

    Returns:
        래퍼가 적용된 Gymnasium 환경
    """
    env = gym.make(env_id)

    # Domain Randomization 래퍼
    if domain_randomization and dr_config:
        env = DomainRandomizationWrapper(
            env,
            friction_range=tuple(dr_config.get("friction_range", [0.7, 1.3])),
            mass_range=tuple(dr_config.get("mass_range", [0.8, 1.2])),
            damping_range=tuple(dr_config.get("damping_range", [0.75, 1.25])),
        )

    # 커스텀 리워드 래퍼
    if custom_reward_fn is not None:
        env = CustomRewardWrapper(env, custom_reward_fn)

    # 관측값 정규화
    if normalize_obs:
        env = NormalizeObservation(env)

    # 보상 정규화 (off-policy 알고리즘에는 replay buffer 비호환으로 비권장)
    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma)

    # 고정 보상 스케일링 (normalize_reward=False일 때 off-policy 알고리즘에 사용)
    if reward_scale != 1.0:
        env = ScaleReward(env, scale=reward_scale)

    return env
