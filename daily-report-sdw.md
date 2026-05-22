# 📝 Daily Report — 동우(A)

> 📌 [README](README.md) · [프로젝트 기획서](project_proposal.md) · [로드맵](roadmap.md)

---

## 🎯 역할 & 진행 상태

| # | 역할 | 상태 | 비고 |
|---|---|---|---|
| 1 | MuJoCo 환경 세팅 | ✅ 완료 | Gymnasium + MuJoCo 구축, CUDA/torch 세팅, seed 고정, 학습 파이프라인 정리 |
| 2 | 공통 학습 프레임워크 제작 | ✅ 완료 | `BaseAlgorithm` 통일 인터페이스, `train.py` 통합 CLI |
| 3 | Evaluation Pipeline 구축 | ✅ 완료 | `scripts/eval_all.py` 작성 완료 |
| 4 | 시각화 자동화 | ✅ 완료 | `scripts/plot_results.py` 작성 완료 |
| 5 | Domain Randomization 환경 구현 | ✅ 완료 | `DomainRandomizationWrapper` 구현됨 |

---

## 📅 작업 내역

### 5/19 (월)

#### 1. 환경 세팅 검증
- Python 가상환경 + `requirements.txt` 설치 확인
- **PyTorch 2.11.0 (CPU 버전)** 확인 — GPU 미탑재 환경
- MuJoCo 3개 환경 정상 동작 확인:
  - `HalfCheetah-v5` — Obs: (17,), Act: (6,) ✅
  - `Ant-v5` — Obs: (105,), Act: (8,) ✅
  - `Humanoid-v5` — Obs: (348,), Act: (17,) ✅

#### 2. 학습 파이프라인 테스트
- PPO × HalfCheetah-v5 × seed42 — 10만 스텝 테스트 실행
- 결과: ~181초(3분), 550 FPS, Return 우상향 확인 → **파이프라인 정상 동작**

#### 3. Evaluation Pipeline 구축 — `scripts/eval_all.py`
- `results/` 하위 실험 폴더를 **자동 탐색** (네이밍 컨벤션으로 파싱)
- 모든 모델을 일괄 평가 → **통합 CSV** (`results/eval/summary.csv`) 출력
- `--filter`, `--episodes` 옵션 지원
- 테스트 실행 완료 (3개 실험 평가 성공)

#### 4. 시각화 자동화 — `scripts/plot_results.py`
- 4종 그래프 자동 생성 스크립트 작성:
  - 학습 곡선 비교 (다중 시드 평균 ± 표준편차)
  - 최종 성능 바 차트 (summary.csv 기반)
  - Reward Shaping 효과 비교
  - Domain Randomization 효과 비교
- 스타일 통일: PPO=파랑, SAC=주황, TD3=초록, 300 DPI
- 테스트 실행 완료 (`curve_HalfCheetah-v5.png`, `performance_bar.png` 생성)

#### 5. 프로젝트 문서 정비
- `project_proposal.md` — 새 역할 분담 반영 (A: 인프라, B: PPO+TD3, C: SAC+보고서)
- `README.md` — 팀원 테이블 업데이트 + 네비게이션 바 추가
- `roadmap.md` — 전체 현황 요약 배치 + 주차별 계획 재작성 + 네비게이션 바 추가
- 3개 문서 네비게이션 양식 통일: `> 📌 [README] · [기획서] · [로드맵]`

---

### 🔜 다음 할 일
- [x] 희승(B), 민설(C)에게 환경 세팅 가이드 공유
- [x] 희승/민설용 Hyperparameter 실험 YAML 프로필 준비 (선택)
- [ ] 팀원들 학습 완료 후 `eval_all.py` → `plot_results.py` 실행하여 최종 결과 수집
