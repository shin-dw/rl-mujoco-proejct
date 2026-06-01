# MuJoCo RL 프로젝트 — State / Action / Reward / Episode 레퍼런스

> 환경: **Humanoid-v5** | 알고리즘: PPO / SAC / TD3  
> 작성일: 2026-06-01

---

## 목차
1. [State (관측 공간)](#1-state-관측-공간)
2. [Action (행동 공간)](#2-action-행동-공간)
3. [Reward (보상 함수)](#3-reward-보상-함수)
4. [Episode (에피소드 구조)](#4-episode-에피소드-구조)
5. [네트워크 구조](#5-네트워크-구조)
6. [파일 구조](#6-파일-구조)

---

## 1. State (관측 공간)

### 1-1. Humanoid-v5 원시 관측값 (348차원)

```
obs[0]   : torso z-height         정상 ~1.2m, 낙하 기준 < 0.7m
obs[1]   : quaternion w
obs[2]   : quaternion x           전후 기울기(pitch) ← uprightness 계산에 사용
obs[3]   : quaternion y           좌우 기울기(roll)  ← uprightness 계산에 사용
obs[4]   : quaternion z
obs[5~]  : 관절 각도 (joint positions)
obs[~]   : 관절 각속도 (joint velocities)
obs[~]   : 발 접촉 정보 (contact forces)
```

### 1-2. NormalizeObservation 래퍼 (`src/common/env_wrapper.py`)

```python
class NormalizeObservation(gym.Wrapper):
    """
    관측값을 Running Mean/Std로 정규화합니다.
    (obs - mean) / std 형태로 정규화 (Welford's online algorithm).
    """
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(obs_shape, dtype=np.float64)
        self.running_var  = np.ones(obs_shape, dtype=np.float64)
        self.count = 0

    def _update_stats(self, obs: np.ndarray):
        self.count += 1
        delta  = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var  += (delta * delta2 - self.running_var) / self.count

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
```

### 1-3. 리워드 함수에서 obs 사용 예시

```python
# next_obs는 NormalizeObservation이 적용된 정규화된 값
qx = float(next_obs[2])   # quaternion x (pitch)
qy = float(next_obs[3])   # quaternion y (roll)

# 직립도 계산: torso_upright ≈ cos(θ), θ = 수직에서 기울어진 각도
torso_upright = 1.0 - 2.0 * (qx**2 + qy**2)
```

### 1-4. 환경 래핑 순서 (`src/common/env_wrapper.py` → `make_env()`)

```
gym.make(env_id)
    └─ DomainRandomizationWrapper  (--domain-rand 옵션 시)
    └─ CustomRewardWrapper         (--reward-type 옵션 시)
    └─ NormalizeObservation        (기본 활성화)
    └─ NormalizeReward             (PPO/SAC/TD3 모두 False — replay buffer 비호환)
    └─ ScaleReward                 (reward_scale != 1.0 시)
```

> ⚠️ **중요**: `CustomRewardWrapper`는 `NormalizeObservation` 안쪽에 있으므로  
> 리워드 함수가 받는 `next_obs`는 **정규화된 값**이다.

---

## 2. Action (행동 공간)

### 2-1. Humanoid-v5 액션 인덱스 (17차원, ctrlrange = ±0.4)

```
[0]  abdomen_yz      복부 비틀림
[1]  abdomen_xz      복부 앞뒤
[2]  abdomen_xy      복부 좌우

[3]  right_hip_x     오른쪽 고관절 — 좌우 벌림 (lateral)
[4]  right_hip_z     오른쪽 고관절 — z축
[5]  right_hip_y  ★  오른쪽 고관절 — 전후 스윙 (sagittal)  ← 달리기 핵심
[6]  right_knee   ★  오른쪽 무릎

[7]  left_hip_x      왼쪽 고관절 — 좌우 벌림 (lateral)
[8]  left_hip_z      왼쪽 고관절 — z축
[9]  left_hip_y   ★  왼쪽 고관절 — 전후 스윙 (sagittal)  ← 달리기 핵심
[10] left_knee    ★  왼쪽 무릎

[11] right_ankle_x
[12] right_ankle_y
[13] left_ankle_x
[14] left_ankle_y

[15] right_shoulder ★  오른팔 스윙  ← 팔 역상 스윙에 사용
[16] left_shoulder  ★  왼팔 스윙   ← 팔 역상 스윙에 사용
```

### 2-2. 알고리즘별 액션 출력 범위

| 알고리즘 | 출력 범위 | 방식 |
|---------|-----------|------|
| **PPO** | **(-∞, +∞)** | Unbounded Gaussian — MuJoCo 내부에서 ±0.4 클리핑 |
| **SAC** | (-0.4, +0.4) | tanh(z) × max_action |
| **TD3** | (-0.4, +0.4) | tanh(z) × max_action + exploration noise |

### 2-3. PPO 액션 처리 (`src/common/networks.py`)

```python
class GaussianActor(nn.Module):
    # PPO 스타일: log_std는 state-independent 학습 파라미터
    self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
    # std = e^(-1) ≈ 0.37  → 액션 경계(±0.4)와 비슷한 스케일로 초반 안정성 확보

    # PPO log_std 상한: std ≤ e^(-0.5) ≈ 0.61 (entropy 폭발 방지)
    self._ppo_log_std_max = -0.5

    def forward(self, obs):
        features = self.shared(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, self._ppo_log_std_max).expand_as(mean)
        return mean, log_std
```

### 2-4. ⚠️ PPO 리워드 함수 내 액션 클리핑 (필수!)

```python
# PPO의 action은 unbounded → CustomRewardWrapper가 raw action을 받음
# 리워드 함수 내에서 반드시 clip 후 사용해야 Reward Hacking 방지
clipped = np.clip(action, -0.4, 0.4)

# ❌ 잘못된 예 (reward hacking 발생)
hip_bonus = -1.0 * (action[5] * action[9])   # action[5]=+22이면 → +484/step 폭발

# ✅ 올바른 예
hip_bonus = -1.0 * (clipped[5] * clipped[9]) # 최대 ±0.4 × ±0.4 = ±0.16
```

---

## 3. Reward (보상 함수)

> 파일: `src/rewards/custom_rewards.py`

### 3-1. Base MuJoCo Reward 구성

```
base_reward = healthy_reward + forward_reward - control_cost
            = 5.0           + (x_vel × weight) - (0.1 × Σ action²)
```

> `base = reward`를 그대로 사용하면 control_cost가 이미 포함된 것임.  
> action 페널티를 추가로 넣으면 **이중 페널티** → 정책이 action=0 학습 → 낙하

### 3-2. `humanoid_balanced_walk` (균형 보행)

```python
def humanoid_balanced_walk(obs, action, reward, next_obs, info) -> float:
    forward_reward = reward                          # base (healthy + forward - control)
    clipped = np.clip(action, -0.4, 0.4)            # PPO unbounded → clip 필수

    # z축 높이 유지
    height = next_obs[0]
    uprightness_bonus = -0.5 * (height - 1.2) ** 2  # 1.2m 기준

    # 에너지 절약 (base의 control_cost와 다른 계수로 추가 억제)
    action_penalty = -0.02 * np.sum(clipped ** 2)

    # x방향 직진 유도
    y_velocity_penalty = -0.3 * abs(info.get("y_velocity", 0.0))

    # 좌우 고관절 대칭 (lateral: [3]=right_hip_x, [7]=left_hip_x)
    symmetry_penalty = -0.1 * abs(clipped[3] - clipped[7])

    return forward_reward + uprightness_bonus + action_penalty + y_velocity_penalty + symmetry_penalty
```

### 3-3. `HumanoidStableGait` (안정적 보행 — 클래스 형태)

```python
class HumanoidStableGait:
    """
    이전 action을 기억해 연속 행동 변화량 페널티 계산.
    get_reward_fn() 호출 시 새 인스턴스 생성 → prev_action 상태 격리됨.
    """
    def __init__(self):
        self.prev_action = None

    def __call__(self, obs, action, reward, next_obs, info) -> float:
        forward_reward = reward
        clipped = np.clip(action, -0.4, 0.4)

        # action 크기 페널티
        smoothness_penalty = -0.01 * np.sum(clipped ** 2)

        # 이전 대비 변화량 페널티 (실제 smoothness)
        if self.prev_action is not None:
            action_diff_penalty = -0.1 * np.sum((clipped - self.prev_action) ** 2)
        else:
            action_diff_penalty = 0.0

        self.prev_action = clipped.copy()
        return forward_reward + smoothness_penalty + action_diff_penalty
```

### 3-4. `humanoid_run_forward` (달리기 특화 — 현재 사용 중)

```python
def humanoid_run_forward(obs, action, reward, next_obs, info) -> float:
    x_vel = info.get("x_velocity", 0.0)
    y_vel = info.get("y_velocity", 0.0)
    base    = reward                           # MuJoCo base reward
    clipped = np.clip(action, -0.4, 0.4)      # PPO unbounded → clip 필수

    # ── 1. 전진 속도 보너스 ───────────────────────────────────────────
    velocity_bonus = 3.0 * max(x_vel, 0.0)

    # ── 2. 직립 보너스 / 기울기 페널티 ──────────────────────────────
    # torso_upright = cos(θ), θ = 수직 기준 기울어진 각도
    # 5.0 × (cos(θ) - cos(30°))  →  0°: +0.67 / 30°: 0.0 / 45°: -0.80 / 60°: -1.83
    qx = float(next_obs[2])
    qy = float(next_obs[3])
    torso_upright    = 1.0 - 2.0 * (qx**2 + qy**2)  # ≈ cos(θ)
    uprightness_bonus = 5.0 * (torso_upright - 0.866)  # 0.866 = cos(30°)

    # ── 3. 고관절 sagittal 교차 보너스 ──────────────────────────────
    # [5]=right_hip_y, [9]=left_hip_y (전후 스윙 주축)
    # 두 값이 반대 방향이면 곱이 음수 → 보너스 양수 (자연스러운 교차 보행)
    hip_alternation_bonus = -1.0 * (clipped[5] * clipped[9])

    # ── 4. 무릎 사용 유도 ────────────────────────────────────────────
    knee_usage_bonus = 0.2 * (abs(clipped[6]) + abs(clipped[10]))

    # ── 5. 팔 역상 스윙 보너스 ───────────────────────────────────────
    # 오른팔[15] ↔ 오른다리[5] 역방향 / 왼팔[16] ↔ 왼다리[9] 역방향
    if len(clipped) > 16:
        arm_swing_bonus = -0.5 * (clipped[15] * clipped[5] + clipped[16] * clipped[9])
    else:
        arm_swing_bonus = 0.0

    # ── 6. Y축 직진 유도 ─────────────────────────────────────────────
    y_penalty = -1.0 * abs(y_vel)

    # ── 7. 셔플링 페널티 (0.5m/s 미만 정체 방지) ─────────────────────
    shuffle_penalty = -2.0 if x_vel < 0.5 else 0.0

    return (base + velocity_bonus + uprightness_bonus
            + hip_alternation_bonus + knee_usage_bonus
            + arm_swing_bonus + y_penalty + shuffle_penalty)
```

### 3-5. 각 항목 기대 범위

| 항목 | 정상 범위 | 비고 |
|------|-----------|------|
| `base` | ~5~8 / step | healthy(5.0) + forward - ctrl_cost |
| `velocity_bonus` | ~0~+9 | x_vel=3m/s → +9 |
| `uprightness_bonus` | -1.83~+0.67 | 직립 +0.67, 60° 기울기 -1.83 |
| `hip_alternation_bonus` | 0~+0.16 | 교차 시 양수, 0.4×0.4=0.16 최대 |
| `knee_usage_bonus` | 0~+0.16 | 두 무릎 합산 최대 |
| `arm_swing_bonus` | 0~+0.16 | 역상 스윙 시 양수 |
| `y_penalty` | ~0~-1 | 직진 시 0 |
| `shuffle_penalty` | 0 or -2.0 | x_vel ≥ 0.5 이면 0 |

### 3-6. 리워드 레지스트리 및 사용법

```python
# src/rewards/custom_rewards.py
REWARD_REGISTRY = {
    "Humanoid-v5": {
        "balanced_walk": humanoid_balanced_walk,   # 함수
        "stable_gait":   HumanoidStableGait,       # 클래스 (상태 유지)
        "run_forward":   humanoid_run_forward,     # 함수
    },
}

def get_reward_fn(env_id: str, reward_type: str):
    fn = REWARD_REGISTRY[env_id][reward_type]
    if isinstance(fn, type):
        return fn()   # 클래스면 인스턴스 생성
    return fn         # 함수면 그대로 반환
```

```bash
# 학습 실행 예시
python -m src.train --algo ppo --env Humanoid-v5 --seed 42 --reward-type run_forward --tensorboard
```

### 3-7. Eval 리워드 주의사항

```
학습 env : gym.make() → CustomRewardWrapper(run_forward) → NormalizeObservation
평가 env : gym.make() → NormalizeObservation  (CustomRewardWrapper 없음!)

→ eval/mean_return은 순수 MuJoCo base reward 기준
   TensorBoard의 train/mean_return과 다른 스케일이 정상
```

---

## 4. Episode (에피소드 구조)

### 4-1. 종료 조건

| 조건 | 플래그 | 원인 |
|------|--------|------|
| torso z-height < 0.7m | `terminated = True` | 낙하 |
| 1000 스텝 도달 | `truncated = True` | 타임아웃 |

### 4-2. PPO 학습 루프 (`src/train.py` → `train_ppo()`)

```python
def train_ppo(algo, env, evaluator, logger, total_steps, eval_freq, log_freq, save_dir):
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    for step in range(1, total_steps + 1):
        algo.anneal_lr(step, total_steps)    # LR 선형 감소 (lr_annealing=True 시)

        action, log_prob, value = algo.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # truncated가 아닌 terminated만 진짜 terminal
        # truncated를 done=1로 저장하면 GAE가 next_value를 0으로 만들어 리턴 과소추정
        algo.buffer.add(obs, action, reward, value, log_prob, terminated)

        ep_ret += reward
        ep_len += 1
        obs = next_obs

        if done:
            logger.log_episode(ep_ret, ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if algo.buffer.full:  # n_steps=2048 도달 시 업데이트
            with torch.no_grad():
                last_val = algo.critic(algo._to_tensor(obs).unsqueeze(0)).item()
            metrics = algo.update(last_val, done)
            ...
```

### 4-3. GAE (Generalized Advantage Estimation) — `src/common/buffer.py`

```python
def compute_gae(self, last_value: float, last_done: bool):
    """
    A_t = Σ (γλ)^l × δ_{t+l}
    δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
    """
    last_gae = 0.0
    for t in reversed(range(self.n_steps)):
        next_value       = last_value if t == self.n_steps - 1 else self.values[t + 1]
        next_non_terminal = 1.0 - self.dones[t]              # terminated면 0, 아니면 1

        delta    = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
        last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        self.advantages[t] = last_gae

    self.returns = self.advantages + self.values              # return = advantage + value
```

### 4-4. PPO 업데이트 주기 요약

```
총 3,000,000 스텝 (Humanoid-v5)
├── 2,048 스텝마다 GAE 계산 → 업데이트 (1회 rollout)
│   └── 10 epochs × (2048 ÷ 512 mini-batch) = 40회 gradient update
├── 50,000 스텝마다 eval (run_forward 실험)
└── 250,000 스텝마다 모델 체크포인트 저장
```

### 4-5. Off-policy (SAC/TD3) 학습 루프 (`train_offpolicy()`)

```python
def train_offpolicy(algo, env, ...):
    for step in range(1, total_steps + 1):
        if step < algo.learning_starts:   # 초반 10,000 스텝: 랜덤 탐험
            action = env.action_space.sample()
        else:
            action = algo.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        algo.buffer.add(obs, action, reward, next_obs, terminated)  # ReplayBuffer

        if step >= algo.learning_starts:
            metrics = algo.update()   # 매 스텝마다 업데이트 (gradient_steps회)
```

---

## 5. 네트워크 구조

### 5-1. 공통 MLP (`src/common/networks.py`)

```
input(obs_dim) → Linear(256) → ReLU → Linear(256) → ReLU → output
```

### 5-2. 알고리즘별 네트워크

| 알고리즘 | Actor | Critic |
|---------|-------|--------|
| **PPO** | GaussianActor (state-independent std) → 출력: mean(17), log_std(17) | VNetwork: obs → V(1) |
| **SAC** | GaussianActor (state-dependent std + tanh + max_action) | TwinQNetwork: [obs+action] → Q1, Q2 (512×512) |
| **TD3** | DeterministicActor: obs → tanh → ×max_action | TwinQNetwork: [obs+action] → Q1, Q2 |

### 5-3. PPO Loss 구성 (`src/algorithms/ppo.py`)

```python
# Clipped Surrogate Objective
ratio  = exp(new_log_probs - old_log_probs)   # 정책 변화 비율
surr1  = ratio * advantages
surr2  = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
policy_loss = -min(surr1, surr2).mean()        # clip_ratio = 0.1 (보수적 업데이트)

# Value Loss
value_loss = MSE(V(s), returns)

# 전체 Loss
loss = policy_loss + 0.5 × value_loss - 0.001 × entropy
```

---

## 6. 파일 구조

```
src/
├── train.py                   # 학습 진입점 (CLI)
├── algorithms/
│   ├── base.py                # BaseAlgorithm (save/load 공통)
│   ├── ppo.py                 # PPO 구현
│   ├── sac.py                 # SAC 구현
│   └── td3.py                 # TD3 구현
├── common/
│   ├── env_wrapper.py         # NormalizeObservation, CustomRewardWrapper, make_env()
│   ├── networks.py            # GaussianActor, DeterministicActor, TwinQNetwork, VNetwork
│   ├── buffer.py              # RolloutBuffer (PPO), ReplayBuffer (SAC/TD3)
│   ├── logger.py              # CSV + TensorBoard 로깅
│   └── evaluator.py           # 학습 중 평가
└── rewards/
    └── custom_rewards.py      # 커스텀 리워드 함수 3종 + REWARD_REGISTRY

configs/
└── default.yaml               # 모든 하이퍼파라미터 (단일 진실 소스)

scripts/
├── run_experiments.py         # 실험 자동화 (baseline → reward → hp → best)
├── eval_all.py                # 전체 모델 일괄 평가 → summary.csv
├── plot_results.py            # 결과 그래프 생성
└── generate_report.py         # 보고서 자동 생성
```

### 주요 하이퍼파라미터 (`configs/default.yaml`)

```yaml
ppo:
  lr_actor: 3.0e-4
  lr_critic: 3.0e-4
  n_steps: 2048          # 롤아웃 크기
  batch_size: 512
  n_epochs: 10
  clip_ratio: 0.1        # 0.2 → 0.1: 보수적 업데이트 (정책 진동 방지)
  gae_lambda: 0.95
  entropy_coef: 0.001
  lr_annealing: true     # LR 선형 감소: lr_init → 0
  reward_scale: 1.0

sac:
  target_entropy: -8.5   # 기본 -17은 std 붕괴 유발 → -8.5로 조정
  reward_scale: 10.0     # Q_max ≈ 60 유지 (Huber 이차 영역)

td3:
  gradient_steps: 8
  reward_scale: 10.0
```
