# ============================================================
# GPU/CPU 겸용 이미지
# - GPU 학습: SAC/TD3 replay buffer 업데이트에 GPU 가속
# - CPU 학습: PPO on-policy 소규모 배치는 CPU로 충분
# PyTorch 이미지 기반 (Python 3.11 + CUDA 12.4 + cuDNN 내장)
# ============================================================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# MuJoCo 헤드리스 렌더링 및 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libosmesa6 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

# PyTorch는 베이스 이미지에 포함 — 나머지 의존성만 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# osmesa: 학습 중 렌더링 비활성화 시에도 MuJoCo 초기화에 필요
ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1
