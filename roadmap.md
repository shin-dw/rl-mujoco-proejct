> 📌 [README](README.md) · [프로젝트 기획서](project_proposal.md) · **로드맵**

# 🗺️ MuJoCo RL 프로젝트 성공 로드맵 (마감: 6/12)

이제 코드는 완성되었습니다. 좋은 성적(A+)을 받기 위해서는 **"얼마나 실험을 체계적으로 했는가"**와 **"결과를 얼마나 예쁘게 보여주는가"**가 핵심입니다. 

팀원 3명이 나누어 진행할 구체적인 플랜입니다.

---

## 📋 전체 현황 요약 (최종 수정: 5/19)

### ✅ 이미 완료된 것 (코드베이스 기준)

| 항목 | 상태 | 비고 |
|---|---|---|
| MuJoCo + Gymnasium 환경 구축 | ✅ 완료 | `env_wrapper.py`에 `make_env()` 팩토리 함수 구현됨 |
| 공통 네트워크 모듈 | ✅ 완료 | `networks.py` — MLP, GaussianActor, DeterministicActor, QNetwork, TwinQ, VNetwork |
| 경험 버퍼 | ✅ 완료 | `buffer.py` — ReplayBuffer (off-policy), RolloutBuffer (on-policy + GAE) |
| 알고리즘 기반 클래스 | ✅ 완료 | `base.py` — 통일 인터페이스 (select_action, update, save, load) |
| PPO / SAC / TD3 구현 | ✅ 완료 | 3개 알고리즘 모두 구현 완료 |
| 통합 학습 스크립트 | ✅ 완료 | `train.py` — CLI 기반, 알고리즘/환경/시드/리워드/DR 옵션 지원 |
| 평가 스크립트 | ✅ 완료 | `evaluate.py` — 모델 로드 → 평가 → CSV 저장 |
| 로거 (CSV + TensorBoard) | ✅ 완료 | `logger.py` — 콘솔/CSV/TensorBoard 통합 |
| Evaluator | ✅ 완료 | `evaluator.py` — 학습 중 주기적 평가 |
| 커스텀 리워드 함수 | ✅ 완료 | `custom_rewards.py` — 6개 리워드 함수 + 레지스트리 |
| Domain Randomization 래퍼 | ✅ 완료 | `env_wrapper.py` — 마찰/질량/감쇠 무작위화 |
| 설정 파일 | ✅ 완료 | `configs/default.yaml` — 공통/PPO/SAC/TD3/DR 설정 |
| Seed 고정 | ✅ 완료 | `base.py`에서 torch/numpy/cuda 시드 고정 |

### ⚠️ 현재 실험 진행 상태

| 실험 | 상태 | 비고 |
|---|---|---|
| `ppo_HalfCheetah-v5_seed42` | ⚠️ 모델만 존재 | `model_final.pt` 있음, 로그 미비 |
| `sac_HalfCheetah-v5_seed42` | ⚠️ 폴더만 존재 | 로그/모델 폴더는 있으나 파일 없음 |
| `td3_HalfCheetah-v5_seed42` | ⚠️ 폴더만 존재 | 로그/모델 폴더는 있으나 파일 없음 |
| `results/eval/` | ❌ 비어있음 | 아직 evaluate.py 실행 안 됨 |

> [!IMPORTANT]
> 코드는 완성되어 있으나, **체계적인 실험 데이터가 거의 없는 상태**입니다.
> 지금부터의 핵심은 **실험 수행 → 데이터 수집 → 시각화 → 보고서 자료 확보**입니다.

---

## 👥 역할 분담

| 역할 | 동우(A) | 희승(B) | 민설(C) |
|---|---|---|---|
| **알고리즘** | — (인프라 전담) | PPO + TD3 구현 | SAC 구현 |
| **공통 인프라** | 환경 세팅, 공통 프레임워크, Eval Pipeline, 시각화 자동화 | Hyperparameter 실험 | 결과 분석 |
| **실험 담당** | Domain Randomization + 기본 성능 비교 | HP 튜닝 + Reward Shaping 일부 | SAC ablation (α 자동조절 등) |
| **보고서** | 서론 + 실험 셋업 | 결과 분석 + 시각화 | 토의 + 결론 + 최종 보고서 |

---

## 📅 주차별 진행 계획

### 📍 1주차 (5/19 ~ 5/25) : 베이스라인(기본 성능) 확보

아무 옵션도 주지 않고 기본 알고리즘이 얼마나 잘 작동하는지 '기준점'을 잡는 주간입니다.

*   **동우(A)**: 환경 세팅 최종 검증 + 희승·민설에게 세팅 가이드 공유 + 베이스라인 실험 관리
*   **희승(B)**: PPO·TD3로 HalfCheetah, Ant, Humanoid 학습 (시드 42, 77, 123)
*   **민설(C)**: SAC로 HalfCheetah, Ant, Humanoid 학습 (시드 42, 77, 123)

> **체크포인트**: TensorBoard를 열었을 때, 시간이 지남에 따라 그래프가 우상향(성능 증가)하는지 확인. (안 올라가면 `--total-steps`를 더 길게 줘서 다시 돌려야 함)

### 📍 2주차 (5/26 ~ 6/1) : 심화 실험 ① - Reward Shaping + HP 튜닝

교수님이 가장 좋아하실 '창의성' 점수를 따는 주간입니다.

*   **동우(A)**: 시각화 자동화 스크립트 작성 (학습 곡선 비교, 성능 바 차트 등)
*   **희승(B)**: Reward Shaping 실험 (`--reward-type energy_efficient`, `stability` 등) + Hyperparameter 민감도 분석 (clip ratio, policy delay, lr 등)
*   **민설(C)**: SAC ablation 실험 (α 자동조절 vs 고정) + 결과 정리 시작

*   **결과 비교**: 기본으로 학습한 로봇과, 커스텀 리워드로 학습한 로봇의 **걷는 폼(자세) 차이**를 메모해 둡니다.

### 📍 3주차 (6/2 ~ 6/8) : 심화 실험 ② - Domain Randomization + 최종 시각화

로봇에게 악조건(돌발 상황)을 부여하는 주간입니다.

*   **동우(A)**: Domain Randomization 실험 실행 (`--domain-rand` 옵션) + 일괄 평가 파이프라인 + 최종 그래프 생성 (300 DPI, 논문 스타일)
*   **희승(B)**: 추가 HP 실험 + 로봇 보행 영상 녹화 (`evaluate.py --render`)
*   **민설(C)**: 전체 결과 분석 + 보고서 초안 작성

*   **평가**: DR로 훈련받은 로봇이, 기본 로봇보다 미끄러운 바닥에서 안 넘어지고 잘 걷는지(`evaluate.py`로 렌더링 확인) 테스트합니다.

### 📍 4주차 (6/9 ~ 6/12) : 최종 보고서 작성 및 마감

*   모은 그래프와 영상을 바탕으로 `보고서.pdf`를 작성합니다.
*   단순히 "PPO가 1등했다"가 아니라, **"왜 SAC는 초반에 빠르지만 TD3가 고점이 더 높은지", "커스텀 리워드를 주었더니 로봇의 걸음걸이가 어떻게 예뻐졌는지"**를 분석해서 적는 것이 핵심입니다.

---

## 💻 학습 시 주의사항 (꿀팁)

> [!WARNING]
> **랜덤 시드(Seed)의 마법**
> 강화학습은 운빨(?)이 심합니다. 똑같은 코드를 돌려도 어떨 때는 천재가 되고 어떨 때는 바보가 됩니다.
> 논문급 결과를 내려면 명령어 끝에 `--seed 42`, `--seed 77`, `--seed 123` 등 **시드를 바꿔가며 최소 3번씩** 돌려서 그 평균값을 그래프로 내야 교수님께 "실험을 체계적으로 잘했네"라는 칭찬을 듣습니다.

> [!TIP]
> **학습은 잘 때 돌려두세요!**
> Humanoid 같은 어려운 환경은 학습에 몇 시간씩 걸립니다. 밤에 컴퓨터를 켜두고 명령어에 `--total-steps 3000000` (300만 번) 처럼 큰 숫자를 주고 주무시는 것을 권장합니다. 결과는 `results/` 폴더에 알아서 저장됩니다.
