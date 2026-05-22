# 🛠️ MuJoCo RL 프로젝트 환경 세팅 가이드

본 문서는 동우(A)가 인프라 세팅 후, 희승(B) 및 민설(C)이 동일한 실험 환경을 구성할 수 있도록 작성된 가이드입니다.

---

## 1. 사전 요구사항
* **OS**: Windows / Linux / macOS (Windows 권장)
* **Python**: `3.10` 또는 `3.11` (가상환경 사용 강력 권장)
* **GPU**: (선택) NVIDIA GPU가 있다면 CUDA 세팅, 없어도 CPU(PyTorch)로 동작 가능합니다. 현재 기본 세팅은 `CPU 버전`을 기준으로 테스트 되었습니다.

## 2. Python 가상환경 구성
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 가상환경을 만들고 활성화합니다.

### Windows (cmd/Powershell)
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. 의존성 패키지 설치
가상환경이 활성화된 상태에서 `requirements.txt`를 통해 필수 패키지를 설치합니다.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **참고**: `requirements.txt`에는 `gymnasium[mujoco]`, `torch`, `numpy`, `matplotlib`, `pandas` 등이 포함되어 있습니다. GPU 환경을 사용할 경우 PyTorch 공식 홈페이지의 안내에 따라 GPU 버전의 torch를 재설치해야 할 수 있습니다.

## 4. 환경 테스트
모든 설치가 끝났다면, 파이프라인이 정상 작동하는지 테스트 스크립트를 실행해 봅니다.

```bash
python src/train.py --algo ppo --env HalfCheetah-v5 --timesteps 1000
```
> 스크립트가 오류 없이 완료되고 짧은 학습이 진행된다면 세팅이 완벽하게 끝난 것입니다.

## 5. (선택) 개인별 실험 프로필 사용법
`configs/` 폴더 내에 개인별(희승, 민설) 실험을 위한 YAML 프로필이 준비되어 있습니다. 
학습을 실행할 때 아래와 같이 적용하세요.

* **희승 (PPO/TD3)**: `python src/train.py --algo ppo --env Ant-v5 --config configs/heeseung_profile.yaml`
* **민설 (SAC)**: `python src/train.py --algo sac --env Humanoid-v5 --config configs/minseol_profile.yaml`

---
> 💡 문의사항이나 세팅 중 에러가 발생하면 동우(A)에게 즉시 슬랙/디스코드로 남겨주세요!
