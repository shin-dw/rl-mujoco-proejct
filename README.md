# 🤖 MuJoCo Humanoid-v5 강화학습 알고리즘 비교 연구

> **서강대학교 강화학습의 기초 프로젝트** | 마감: 2026년 6월 12일 (금)

MuJoCo **Humanoid-v5** 환경에서 **PPO, SAC, TD3** 세 알고리즘을 직접 구현하고,  
Reward Shaping 및 Hyperparameter 튜닝 실험을 통해 체계적으로 성능을 비교·분석합니다.

---

## 🏆 최종 실험 결과 요약

> 모든 실험은 **Humanoid-v5** 환경, **seed 42**, **20M 학습 스텝** 기준으로 수행되었습니다.

### 알고리즘별 Eval Return (10 에피소드 평균)

| 알고리즘              | Mean      | Std   | Peak      | 후반 20% 평균 | 비고                |
| --------------------- | --------- | ----- | --------- | ------------- | ------------------- |
| **PPO + run_forward** | **7,251** | 2,076 | **9,378** | **6,777**     | 최고 성능           |
| TD3 Baseline          | 6,164     | 1,917 | 7,429     | 5,403         | 학습 후반 불안정    |
| PPO Baseline          | 4,176     | 1,259 | 5,507     | 4,966         | 가장 안정적         |
| SAC Baseline          | 2,415     | 1,577 | 4,998     | 3,044         | reward scaling 이슈 |

### 핵심 발견

- **Reward Shaping 효과**: PPO Baseline(4,176) → PPO run_forward(7,251), **+73.6% 성능 향상**
- **알고리즘 안정성**: PPO가 가장 안정적 (Std 최소), TD3는 후반부 성능 저하 관찰
- **SAC 한계**: reward_scale 이중 적용(×0.01 효과)으로 학습 신호 약화, 잠재 성능 미달

---

## 📁 프로젝트 구조

```
rl-mujoco-proejct/
├── configs/
│   ├── default.yaml                         # 기본 하이퍼파라미터
│   └── hp_tuning/                           # HP 튜닝용 yaml (12개)
│       ├── ppo_clip01.yaml / ppo_clip03.yaml
│       ├── ppo_lr1e-3.yaml / ppo_lr1e-4.yaml
│       ├── sac_bs128.yaml  / sac_bs512.yaml
│       ├── sac_lra1e-3.yaml/ sac_lra1e-4.yaml
│       ├── td3_delay1.yaml / td3_delay4.yaml
│       └── td3_noise005.yaml/ td3_noise02.yaml
├── src/
│   ├── algorithms/
│   │   ├── base.py                          # BaseAlgorithm 추상 클래스
│   │   ├── ppo.py                           # PPO (Clipped Surrogate + GAE)
│   │   ├── sac.py                           # SAC (Max-Entropy + Twin-Q + Auto-α)
│   │   └── td3.py                           # TD3 (Delayed Update + Target Smoothing)
│   ├── common/
│   │   ├── buffer.py                        # RolloutBuffer (PPO) / ReplayBuffer (SAC, TD3)
│   │   ├── env_wrapper.py                   # NormalizeObs / ScaleReward / DR 래퍼
│   │   ├── evaluator.py                     # 학습 중 정책 평가
│   │   ├── logger.py                        # CSV + TensorBoard 로깅
│   │   └── networks.py                      # MLP 기반 Actor / Critic 네트워크
│   ├── rewards/
│   │   └── custom_rewards.py                # Humanoid-v5 전용 커스텀 리워드
│   ├── train.py                             # 학습 진입점
│   └── evaluate.py                          # 단일 모델 평가 / 렌더링
├── scripts/
│   ├── run_experiments.py                   # 전체 실험 자동화 (baseline→reward→hp→best)
│   ├── eval_all.py                          # 모든 모델 일괄 평가 → summary.csv
│   ├── plot_comparison.py                   # 4개 알고리즘 비교 그래프
│   ├── plot_individual.py                   # 실험별 개별 상세 그래프
│   ├── plot_results.py                      # 종합 결과 그래프
│   └── record_videos.py                     # 에피소드 MP4 녹화
├── results/
│   ├── eval/
│   │   └── summary.csv                      # eval_all.py 일괄 평가 결과 (Git 포함)
│   ├── logs/                                # 학습 로그 (Git 제외)
│   │   └── {algo}_{env}_{tag}_seed{seed}/
│   │       └── logs/
│   │           ├── progress.csv             # 학습 메트릭 (step, return, loss, ...)
│   │           └── tensorboard/             # TensorBoard 이벤트
│   ├── models/                              # 학습된 모델 (Git 포함)
│   │   └── {algo}_{env}_{tag}_seed{seed}/
│   │       └── models/
│   │           ├── model_final.pt           # 최종 모델 가중치
│   │           └── model_final_obs_stats.npz# 관측값 정규화 통계
│   ├── plots/                               # 그래프 이미지 (Git 제외)
│   │   ├── comparison_main.png
│   │   ├── comparison_bar.png
│   │   ├── final_performance.png
│   │   └── individual/{exp}.png
│   └── videos/                              # 에피소드 녹화 영상 (Git 포함)
│       └── {algo}_{env}_{tag}_seed{seed}/
│           └── ep{N}.mp4
├── RL 과제 보고서_제출용.pptx
├── RL_project_handout.pdf
├── requirements.txt
└── README.md
```

---

## 📥 사전 학습된 모델

> 학습된 모델은 리포지토리 `results/models/` 폴더에 포함되어 있습니다.  
> `git clone` 후 별도 설치 없이 바로 평가·렌더링이 가능합니다.

| 경로                                                                                                                                               | 알고리즘          | Mean  | Peak  | 크기   |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----- | ----- | ------ |
| [results/models/ppo_Humanoid-v5_run_forward_seed42/models/model_final.pt](results/models/ppo_Humanoid-v5_run_forward_seed42/models/model_final.pt) | PPO + run_forward | 7,251 | 9,378 | 3.6 MB |
| [results/models/ppo_Humanoid-v5_seed42/models/model_final.pt](results/models/ppo_Humanoid-v5_seed42/models/model_final.pt)                         | PPO Baseline      | 4,176 | 5,507 | 3.6 MB |
| [results/models/sac_Humanoid-v5_seed42/models/model_final.pt](results/models/sac_Humanoid-v5_seed42/models/model_final.pt)                         | SAC Baseline      | 2,415 | 4,998 | 6.8 MB |
| [results/models/td3_Humanoid-v5_seed42/models/model_final.pt](results/models/td3_Humanoid-v5_seed42/models/model_final.pt)                         | TD3 Baseline      | 6,164 | 7,429 | 7.3 MB |

### 클론 후 즉시 평가

```bash
git clone https://github.com/shin-dw/rl-mujoco-proejct.git
cd rl-mujoco-proejct
pip install -r requirements.txt

# 최고 성능 모델 (PPO + run_forward) 렌더링
python -m src.evaluate \
    --model results/models/ppo_Humanoid-v5_run_forward_seed42/models/model_final.pt \
    --algo ppo --env Humanoid-v5 --render --episodes 3

# 알고리즘 비교 평가 (수치 출력)
python -m src.evaluate --model results/models/ppo_Humanoid-v5_seed42/models/model_final.pt   --algo ppo --env Humanoid-v5
python -m src.evaluate --model results/models/sac_Humanoid-v5_seed42/models/model_final.pt   --algo sac --env Humanoid-v5
python -m src.evaluate --model results/models/td3_Humanoid-v5_seed42/models/model_final.pt   --algo td3 --env Humanoid-v5
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상 환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate             # Windows
# source venv/bin/activate        # Linux/Mac

# 의존성 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 12.4 기준 — 권장)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**시스템 요구사항**

- Python 3.10 ~ 3.12
- MuJoCo 3.x (gymnasium[mujoco] 설치 시 자동 포함)
- GPU: CUDA 지원 GPU 권장 (RTX 4060 기준 20M 스텝 ≈ 수 시간)

### 2. 기본 학습 실행

```bash
# PPO Baseline — Humanoid-v5 (기본 seed 42)
python -m src.train --algo ppo --env Humanoid-v5 --seed 42 --tensorboard

# SAC Baseline
python -m src.train --algo sac --env Humanoid-v5 --seed 42 --tensorboard

# TD3 Baseline
python -m src.train --algo td3 --env Humanoid-v5 --seed 42 --tensorboard

# PPO + run_forward Reward Shaping (최고 성능 조합)
python -m src.train --algo ppo --env Humanoid-v5 --seed 42 \
    --reward-type run_forward --tensorboard
```

#### 주요 옵션

| 옵션            | 설명                    | 예시                                          |
| --------------- | ----------------------- | --------------------------------------------- |
| `--algo`        | 알고리즘 선택 (필수)    | `ppo`, `sac`, `td3`                           |
| `--env`         | MuJoCo 환경             | `Humanoid-v5`                                 |
| `--seed`        | 랜덤 시드               | `42`                                          |
| `--total-steps` | 총 학습 스텝            | `20000000`                                    |
| `--reward-type` | 커스텀 리워드 함수      | `run_forward`, `balanced_walk`, `stable_gait` |
| `--config`      | 설정 파일 경로          | `configs/hp_tuning/ppo_clip01.yaml`           |
| `--tensorboard` | TensorBoard 로깅 활성화 | (플래그)                                      |

### 3. TensorBoard로 학습 모니터링

```bash
# 단일 실험
tensorboard --logdir results/logs/ppo_Humanoid-v5_seed42/logs/tensorboard

# 전체 실험 비교
tensorboard --logdir results/logs/
# → 브라우저에서 http://localhost:6006 접속
```

### 4. 학습된 모델 평가

```bash
# 숫자 결과만 출력
python -m src.evaluate \
    --model results/models/ppo_Humanoid-v5_run_forward_seed42/models/model_final.pt \
    --algo ppo --env Humanoid-v5

# 시뮬레이션 창에서 렌더링
python -m src.evaluate \
    --model results/models/ppo_Humanoid-v5_run_forward_seed42/models/model_final.pt \
    --algo ppo --env Humanoid-v5 --render --episodes 3
```

### 5. 전체 실험 자동화

```bash
# 모든 단계 순서대로 실행 (baseline → reward shaping → HP tuning → best)
python scripts/run_experiments.py

# 단계별 개별 실행
python scripts/run_experiments.py --phase baseline   # 3개 알고리즘 기본 학습
python scripts/run_experiments.py --phase reward     # Reward Shaping 실험

# 최적 조합 탐색 (사전에 eval_all.py 실행 필수)
python scripts/eval_all.py
python scripts/run_experiments.py --phase best
```

---

## 🧪 실험 설계

### 대상 환경

| 환경            | State 차원 | Action 차원 | Action 범위 | 학습 스텝  |
| --------------- | ---------- | ----------- | ----------- | ---------- |
| **Humanoid-v5** | 376        | 17          | ±0.4        | 20,000,000 |

> MuJoCo 인간형 로봇의 직립 보행 학습. 가장 고차원 연속 제어 환경.

### 구현된 알고리즘

| 알고리즘 | 방식       | 버퍼                | 특징                                                         |
| -------- | ---------- | ------------------- | ------------------------------------------------------------ |
| **PPO**  | On-policy  | RolloutBuffer (GAE) | clip_ratio=0.1, lr annealing, state-independent std          |
| **SAC**  | Off-policy | ReplayBuffer        | Twin-Q, Auto-α, target_entropy=-8.5                          |
| **TD3**  | Off-policy | ReplayBuffer        | Delayed update (×2), target smoothing, exploration noise=0.1 |

### 커스텀 리워드 (Humanoid-v5 전용)

| 리워드 타입     | 설명                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `run_forward`   | velocity_bonus(×3.0) + 자세 페널티 + 엉덩이 교번 보너스 + 무릎/팔 스윙 + y축 직진 + 제자리 걸음 페널티 |
| `balanced_walk` | 기본 reward + 높이 유지 + 에너지 절약 + y축 직진 + 좌우 고관절 대칭성                                  |
| `stable_gait`   | 연속 행동 변화량 페널티 (부드러운 걸음걸이)                                                            |

---

## 📊 결과 분석 및 시각화

### 그래프 생성

```bash
# 4개 알고리즘 비교 그래프 (comparison_main.png, comparison_bar.png)
python scripts/plot_comparison.py

# 실험별 개별 상세 그래프 (progress.csv 기반)
python scripts/plot_individual.py

# 전체 실험 종합 그래프
python scripts/plot_results.py --env Humanoid-v5

# 특정 알고리즘만 필터링
python scripts/plot_individual.py --filter ppo
python scripts/plot_individual.py --filter hp_clip
```

저장 위치: `results/plots/`

### 전체 평가 실행

```bash
# 모든 학습된 모델을 일괄 평가 → results/eval/summary.csv 생성
python scripts/eval_all.py
```

### 에피소드 영상 녹화

```bash
# 4개 주요 실험 녹화 (각 10 에피소드)
python scripts/record_videos.py

# 에피소드 수 조정
python scripts/record_videos.py --episodes 3

# 특정 알고리즘만
python scripts/record_videos.py --filter ppo

# 저장 위치: results/videos/{실험명}/ep01.mp4, ep02.mp4, ...
```

> **사전 설치 필요**: `pip install imageio imageio-ffmpeg`

---

## ⚙️ 주요 하이퍼파라미터 (default.yaml)

### PPO

| 파라미터             | 값   | 설명               |
| -------------------- | ---- | ------------------ |
| lr_actor / lr_critic | 3e-4 | 학습률             |
| n_steps              | 2048 | 롤아웃 스텝 수     |
| batch_size           | 512  | 미니배치 크기      |
| n_epochs             | 10   | 업데이트 반복 횟수 |
| clip_ratio           | 0.1  | PPO 클리핑 범위    |
| gae_lambda           | 0.95 | GAE λ              |
| lr_annealing         | true | 학습률 선형 감소   |

### SAC

| 파라미터       | 값   | 설명                                     |
| -------------- | ---- | ---------------------------------------- |
| lr_actor       | 3e-4 | Actor 학습률                             |
| lr_critic      | 1e-4 | Critic 학습률                            |
| batch_size     | 256  | 배치 크기                                |
| target_entropy | -8.5 | Humanoid-v5 최적값 (-17은 std 붕괴 유발) |
| reward_scale   | 10.0 | 리워드 스케일링                          |
| auto_entropy   | true | α 자동 조절                              |

### TD3

| 파라미터          | 값   | 설명                     |
| ----------------- | ---- | ------------------------ |
| lr_actor          | 3e-4 | Actor 학습률             |
| lr_critic         | 1e-4 | Critic 학습률            |
| batch_size        | 1024 | 배치 크기                |
| policy_delay      | 2    | 정책 업데이트 지연 주기  |
| exploration_noise | 0.1  | 탐색 노이즈 (가우시안 σ) |
| reward_scale      | 10.0 | 리워드 스케일링          |

---

## 🔍 실험 범위 및 한계

### 실제 수행된 실험

- ✅ Humanoid-v5 환경, seed 42, 20M 스텝
- ✅ PPO / SAC / TD3 Baseline 학습
- ✅ PPO run_forward Reward Shaping (+73.6% 성능 향상)
- ✅ 알고리즘 간 Eval Return 비교 분석

---

## 👥 팀원

| 이름           | 담당                                                  |
| -------------- | ----------------------------------------------------- |
| 신동우(A74041) | 환경/실험 인프라, 시각화 자동화                       |
| 이희승(A74048) | PPO + TD3 구현, Reward Shaping 실험, 전체 실험 진행행 |
| 박민설(A74038) | SAC 구현, 분석/보고서                                 |

---

## 📝 라이선스

본 프로젝트는 서강대학교 강화학습 수업 과제로 제작되었습니다.
