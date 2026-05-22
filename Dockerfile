FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# MuJoCo 헤드리스 실행에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 디스플레이 없는 환경에서 MuJoCo 렌더링 백엔드
ENV MUJOCO_GL=egl
