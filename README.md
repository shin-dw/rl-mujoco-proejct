# 🤖 MuJoCo 연속 제어 환경에서의 강화학습 알고리즘 비교 연구

> **서강대학교 강화학습 프로젝트**
> 마감: 2026년 6월 12일 (금)

> 📌 **README** · [프로젝트 기획서](project_proposal.md) · [로드맵](roadmap.md)

MuJoCo 물리 시뮬레이션 기반 연속 제어 환경에서 **PPO, SAC, TD3** 알고리즘을 직접 구현하고, Reward Shaping 및 Domain Randomization 실험을 통해 체계적으로 비교·분석합니다.

## 📚 문서 안내

원하시는 목적에 따라 아래의 문서를 확인해 주세요.
*   [**실행 방법 및 환경 설정 (`README.md`)**](./README.md) (현재 문서): 코드 실행 및 결과 확인 방법
*   [**프로젝트 기획서 (`project_proposal.md`)**](./project_proposal.md): 연구 목표, 실험 설계 및 세부 기획 내용
*   [**실행 로드맵 (`roadmap.md`)**](./roadmap.md) (내부용): 팀원 주차별 액션 플랜 및 실험 팁

## 📁 프로젝트 구조

```
rl/
├── configs/
│   └── default.yaml              # Hyperparameter 설정
├── src/
│   ├── common/                   # 공통 모듈
│   │   ├── networks.py           # 신경망 (Actor, Critic, MLP)
│   │   ├── buffer.py             # 경험 버퍼 (Replay, Rollout)
│   │   ├── env_wrapper.py        # 환경 래퍼 (정규화, DR, 커스텀 리워드)
│   │   ├── logger.py             # 학습 로깅 (CSV, TensorBoard)
│   │   └── evaluator.py          # 정책 평가
│   ├── algorithms/               # 알고리즘
│   │   ├── base.py               # 추상 기반 클래스
│   │   ├── ppo.py                # PPO (Proximal Policy Optimization)
│   │   ├── sac.py                # SAC (Soft Actor-Critic)
│   │   └── td3.py                # TD3 (Twin Delayed DDPG)
│   ├── rewards/                  # 커스텀 리워드
│   │   └── custom_rewards.py     # 환경별 리워드 함수 레지스트리
│   ├── train.py                  # 통합 학습 스크립트
│   └── evaluate.py               # 평가 및 시각화 스크립트
├── results/                      # 실험 결과 (자동 생성, Git 제외)
├── project_proposal.md           # 프로젝트 기획서
├── .gitignore
├── requirements.txt              # Python 의존성
└── README.md
```

## 🚀 빠른 시작

### 0. 필수 조건

- **Python 3.13 ~ 3.14** (3.15 pre-release는 비권장)
- GPU가 있으면 학습이 빨라지지만, CPU만으로도 충분히 실행 가능

### 1. 환경 설정

```bash
# 가상 환경 생성 (권장)
python -m venv venv
venv\Scripts\activate             # Windows
# source venv/bin/activate        # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 학습 실행

```bash
# PPO로 HalfCheetah 학습 (기본)
python -m src.train --algo ppo --env HalfCheetah-v5 --seed 42

# SAC로 Ant 학습 (커스텀 리워드 적용)
python -m src.train --algo sac --env Ant-v5 --seed 42 --reward-type energy_efficient

# TD3로 HalfCheetah 학습 (Domain Randomization 적용)
python -m src.train --algo td3 --env HalfCheetah-v5 --seed 42 --domain-rand

# TensorBoard 로깅 활성화 (학습 그래프 실시간 확인)
python -m src.train --algo sac --env HalfCheetah-v5 --seed 42 --tensorboard
```

#### 주요 옵션 설명

| 옵션 | 설명 | 예시 |
|---|---|---|
| `--algo` | 사용할 알고리즘 (필수) | `ppo`, `sac`, `td3` |
| `--env` | MuJoCo 환경 | `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` |
| `--seed` | 랜덤 시드 (재현성 보장) | `42`, `77`, `123` |
| `--total-steps` | 총 학습 스텝 수 | `1000000` (기본: 환경별 자동 설정) |
| `--reward-type` | 커스텀 리워드 함수 | `energy_efficient`, `stability` |
| `--domain-rand` | Domain Randomization 활성화 | (플래그, 값 불필요) |
| `--tensorboard` | TensorBoard 로깅 활성화 | (플래그, 값 불필요) |

### 3. 모델 평가 및 시각화

```bash
# 학습된 모델 평가 (숫자 결과만 확인)
python -m src.evaluate --model results\ppo_HalfCheetah-v5_seed42\models\model_final.pt --algo ppo --env HalfCheetah-v5

# 로봇이 걷는 모습을 화면에 렌더링 (MuJoCo 시뮬레이션 창 표시)
python -m src.evaluate --model results\ppo_HalfCheetah-v5_seed42\models\model_final.pt --algo ppo --env HalfCheetah-v5 --render --episodes 3
```

### 4. TensorBoard로 학습 그래프 확인

```bash
tensorboard --logdir results/
# 브라우저에서 http://localhost:6006 접속
```

## 🧪 실험 구성

### 대상 환경

| 환경 | State 차원 | Action 차원 | 난이도 | 권장 학습 스텝 |
|---|---|---|---|---|
| HalfCheetah-v5 | 17 | 6 | ⭐⭐ | 1,000,000 |
| Ant-v5 | 27 | 8 | ⭐⭐⭐ | 2,000,000 |
| Humanoid-v5 | 376 | 17 | ⭐⭐⭐⭐ | 3,000,000 |

### 알고리즘

- **PPO**: On-policy, Clipped Surrogate + GAE
- **SAC**: Off-policy, Maximum Entropy + Twin Q + Auto α
- **TD3**: Off-policy, Deterministic Policy + Delayed Update + Twin Q

### 실험 종류

1. **기본 성능 비교**: 3 알고리즘 × 3 환경 × 3+ 시드
2. **Reward Shaping**: 커스텀 리워드 함수별 학습 효율 비교
3. **Domain Randomization**: 물리 파라미터 무작위화 후 로버스트니스 평가

### 사용 가능한 커스텀 리워드

| 환경 | 리워드 타입 | 설명 |
|---|---|---|
| HalfCheetah-v5 | `energy_efficient` | 에너지 효율 극대화 |
| HalfCheetah-v5 | `stability` | 자세 안정성 중심 |
| Ant-v5 | `directional` | 직진성 보상 |
| Ant-v5 | `energy_efficient` | 에너지 효율 극대화 |
| Humanoid-v5 | `balanced_walk` | 균형 잡힌 보행 |
| Humanoid-v5 | `stable_gait` | 부드러운 걸음걸이 |

## 👥 팀원

| 이름 | 학번 | 담당 |
|---|---|---|
| 동우 | - | 환경/실험 인프라 + Domain Randomization + 시각화 자동화 |
| 희승 | - | PPO + TD3 구현 + Hyperparameter 실험 + Reward Shaping |
| 민설 | - | SAC 구현 + 분석/보고서 |

## 📝 라이선스

본 프로젝트는 서강대학교 강화학습 수업 과제로 제작되었습니다.
