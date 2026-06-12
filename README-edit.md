# MuJoCo 연속 제어 환경에서의 강화학습 알고리즘 비교 및 Reward Engineering

서강대학교 강화학습 프로젝트 (인공지능)

| 이름 | 학번 | 담당 |
|---|---|---|
| 박민설 | A74038 | SAC 구현, 전체 결과 분석, 최종 보고서 작성 |
| 신동우 | A74041 | MuJoCo 환경 세팅, 공통 학습 프레임워크, 시각화 자동화 |
| 이희승 | A74048 | PPO·TD3 구현, Hyperparameter 실험, Reward Shaping |

- 보고서: [`RL 과제 보고서_제출용.pptx`](./RL%20과제%20보고서_제출용.pptx)
- 학습된 최종 모델: [`final/model_final.pt`](./final/) (PPO + run_forward, 20M steps)

## 개요

MuJoCo **Humanoid-v5** (인간형 로봇 보행 제어) 환경에서 **PPO, SAC, TD3**를 외부 RL 라이브러리 없이 PyTorch로 직접 구현하고, 다음을 검증했다.

1. 세 알고리즘의 학습 수렴 속도·최종 성능·안정성 비교
2. 달리기 특화 커스텀 리워드(`run_forward`) 설계를 통한 Reward Shaping 효과 검증
3. 하이퍼파라미터 변경(clip ratio, learning rate, batch size 등)에 따른 성능 비교

## 주요 결과

Humanoid-v5, 20M steps, seed 42. Final Performance는 학습 후반 20% 구간(16M→20M)의 Eval Return 평균(에피소드 10개).

| 알고리즘 | Mean Return | Peak | 후반 20% | 후반 10% | 비고 |
|---|---:|---:|---:|---:|---|
| **PPO + run_forward** | **7,251** | **9,378** | **6,777** | 6,218 | 가장 사람다운 달리기 동작 |
| TD3 | 6,164 | 7,429 | 5,403 | 3,527 | 말기 성능 급락 (불안정) |
| PPO Baseline | 4,176 | 5,507 | 4,966 | 4,994 | 안정적이나 생존 위주 보행 |
| SAC | 2,415 | 4,998 | 3,044 | 3,026 | 고차원 행동 공간에서 학습 불안정 |

- **Reward Shaping 효과**: PPO Baseline 대비 Mean +73%, Peak +70%. 알고리즘 교체보다 리워드 설계가 더 큰 성능 향상을 가져왔다.
- **On-policy vs Off-policy**: 이 환경에서는 PPO(on-policy)가 롤아웃 기반의 안정적 수렴을 보인 반면, TD3(off-policy)는 초반 학습은 빠르나 말기 붕괴 위험이 있었다.
- 상세 분석과 시연 영상은 보고서 6~7장 참고.

## 프로젝트 구조

```
rl/
├── configs/
│   ├── default.yaml            # 알고리즘별 하이퍼파라미터 (최종 설정)
│   └── hp_tuning/              # HP 튜닝 실험용 변형 설정 (12종)
├── src/
│   ├── algorithms/             # PPO / SAC / TD3 직접 구현 (base.py: 공통 추상 클래스)
│   ├── common/
│   │   ├── networks.py         # GaussianActor, DeterministicActor, TwinQ, V (MLP 256×256)
│   │   ├── buffer.py           # RolloutBuffer(PPO), ReplayBuffer(SAC/TD3)
│   │   ├── env_wrapper.py      # Obs 정규화(Welford), Domain Randomization, 커스텀 리워드 래퍼
│   │   ├── evaluator.py        # 학습 중 주기적 평가
│   │   └── logger.py           # CSV + TensorBoard 로깅
│   ├── rewards/custom_rewards.py  # Humanoid-v5 커스텀 리워드 (run_forward 등)
│   ├── train.py                # 통합 학습 스크립트
│   └── evaluate.py             # 모델 평가·렌더링
├── scripts/
│   ├── run_experiments.py      # 실험 자동화 (baseline / reward / hp / best)
│   ├── eval_all.py             # 전체 모델 일괄 평가 → summary.csv
│   ├── plot_results.py         # 보고서용 그래프 자동 생성
│   └── generate_report.py      # 결과 요약 문서 생성
├── final/                      # 최종 모델 가중치 + obs 통계 + 학습 로그
├── results/                    # 실험별 모델·로그 (학습 시 자동 생성)
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## 설치

Python 3.10+ 기준.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

주요 의존성: PyTorch 2.x, Gymnasium 1.x (`gymnasium[mujoco]`), MuJoCo 3.x, NumPy, Matplotlib, TensorBoard

## 실행 방법

### 학습

```bash
# PPO Baseline (Humanoid-v5, 20M steps — configs/default.yaml 기준)
python -m src.train --algo ppo --env Humanoid-v5 --seed 42

# PPO + run_forward 리워드 셰이핑 (최종 최고 성능 조합)
python -m src.train --algo ppo --env Humanoid-v5 --seed 42 --reward-type run_forward

# SAC / TD3
python -m src.train --algo sac --env Humanoid-v5 --seed 42
python -m src.train --algo td3 --env Humanoid-v5 --seed 42

# TensorBoard 로깅 활성화
python -m src.train --algo ppo --env Humanoid-v5 --seed 42 --tensorboard
```

| 옵션 | 설명 |
|---|---|
| `--algo` | `ppo` / `sac` / `td3` (필수) |
| `--env` | MuJoCo 환경 ID (기본 실험: `Humanoid-v5`) |
| `--seed` | 랜덤 시드 |
| `--total-steps` | 총 학습 스텝 (생략 시 config의 환경별 값 사용) |
| `--reward-type` | 커스텀 리워드 (`run_forward`, `balanced_walk`, `stable_gait`) |
| `--config` | 설정 파일 경로 (기본 `configs/default.yaml`) |
| `--device` | `auto` / `cpu` / `cuda` |
| `--domain-rand` | 물리 파라미터 무작위화 (마찰·질량·감쇠) |

결과는 `results/<algo>_<env>[_reward]_seed<seed>/` 아래에 모델(`models/`)과 로그(`logs/progress.csv`)로 저장된다.

### 평가 및 렌더링

```bash
# 최종 모델 평가 (obs 정규화 통계는 모델 옆 _obs_stats.npz에서 자동 복원)
python -m src.evaluate --model final/model_final.pt --algo ppo --env Humanoid-v5 --episodes 10

# 달리는 모습 렌더링
python -m src.evaluate --model final/model_final.pt --algo ppo --env Humanoid-v5 --render --episodes 3
```

### 실험 자동화·시각화

```bash
python scripts/run_experiments.py --phase baseline   # 알고리즘별 베이스라인
python scripts/run_experiments.py --phase reward     # Reward Shaping 비교
python scripts/run_experiments.py --phase hp         # HP 튜닝 (configs/hp_tuning/)
python scripts/eval_all.py --episodes 20             # 전체 모델 일괄 평가
python scripts/plot_results.py --env Humanoid-v5     # 학습 곡선·바 차트 생성
tensorboard --logdir results/                        # http://localhost:6006
```

### Docker

GPU/CPU 겸용 이미지(PyTorch 2.4 + CUDA 12.4). 알고리즘별 동시 학습 등 세부 서비스는 `docker-compose.yml` 주석 참고.

```bash
docker compose run --rm train-ppo-run-forward   # 최종 조합 학습
docker compose up tensorboard
```

## 설계 요약

### State / Action

- **State (348-dim)**: torso 높이, 몸체 quaternion, 관절 각도/각속도, 접촉력. Welford 알고리즘 기반 running mean/std로 정규화하며, 통계는 모델과 함께 `.npz`로 저장되어 평가 시 그대로 복원된다.
- **Action (17-dim)**: 각 관절 actuator의 토크, 범위 [-0.4, 0.4]. PPO는 가우시안 정책 출력(MuJoCo 내부 클리핑), SAC/TD3는 tanh squashing으로 범위를 맞춘다.
- **에피소드 종료**: torso 높이 < 0.7m → terminated(낙상), 1000 스텝 → truncated(타임아웃). PPO의 GAE 계산에서 둘을 구분해 truncated 시 부트스트래핑을 유지한다.

### Reward 설계 (`run_forward`)

MuJoCo 기본 리워드(생존 + 전진 - 제어비용 - 접촉비용)만으로는 "넘어지지 않고 전진"하는 비효율적 보행에 머물러, 달리기 특성을 반영한 항목을 추가했다 (`src/rewards/custom_rewards.py`).

| 항목 | 목적 |
|---|---|
| Velocity Bonus (3.0 × x속도) | 빠른 전진 유도 |
| Posture Penalty | 후방 기울임 강한 페널티, 과도한 전방 숙임 방지 |
| Hip Alternation | 좌우 다리 교차 보행 보상, 양발 동시 점프(캥거루) 강한 페널티 |
| Knee Usage Bonus | 적극적인 무릎 활용 |
| Arm Swing Bonus | 다리와 반대 방향 팔 스윙 유도 |
| Y-axis Penalty | 직진 보행 유도 |
| Shuffle Penalty | 제자리걸음(x속도 < 0.5) 방지 |

### 알고리즘 및 하이퍼파라미터

세 알고리즘 모두 동일한 MLP backbone(Linear 256 → ReLU → Linear 256 → ReLU)을 사용한다. 전체 값은 `configs/default.yaml` 참고.

| | PPO | SAC | TD3 |
|---|---|---|---|
| LR (actor / critic) | 3e-4 / 3e-4 | 3e-4 / 1e-4 (α: 3e-4) | 3e-4 / 1e-4 |
| Batch Size | 512 (rollout 2048) | 256 | 1024 |
| 핵심 설정 | clip 0.1, GAE λ 0.95, epochs 10, entropy 0.001, LR annealing | buffer 1M, auto entropy (target -8.5), τ 0.005 | buffer 1M, policy delay 2, target noise 0.2, gradient steps 8 |
| Reward Scale | 1.0 | 10.0 | 10.0 |

SAC/TD3는 running reward normalization이 replay buffer와 비호환이라 고정 스케일링(÷10)을 적용했고, SAC의 target entropy는 기본값(-17)이 정책 표준편차 붕괴를 유발해 Humanoid-v5 전용으로 -8.5를 사용했다.

## 실험 셋업

- 메인 비교 4종(PPO / SAC / TD3 / PPO+run_forward): Humanoid-v5 × 20M steps × seed 42
- HP 튜닝·Reward Shaping 보조 실험: 3M steps (`configs/hp_tuning/`)
- 평가: 50k 스텝마다 별도 평가 환경에서 10 에피소드, `eval/mean_return` 기록

## 한계 및 향후 연구

- 메인 실험이 단일 시드(42)로 수행되어 통계적 신뢰도 검증이 부족하다. 다중 시드(42/77/123) 실험과 평균±표준편차 신뢰구간 제시가 필요하다.
- Domain Randomization 래퍼는 구현되어 있으나 본 실험에서는 미수행 — 마찰/질량/감쇠 변동 환경에서의 로버스트니스 비교가 다음 과제다.
- 메인 실험(20M)과 보조 실험(3M)의 학습 스텝 불균형으로 공정한 비교에 제한이 있다.
