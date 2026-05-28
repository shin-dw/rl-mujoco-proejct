"""
학습 로깅 모듈

학습 과정의 메트릭을 CSV 파일과 콘솔에 기록합니다.
선택적으로 TensorBoard 로깅도 지원합니다.
"""

import os
import csv
import time
from typing import Dict, Any, Optional
from collections import defaultdict


class Logger:
    """
    학습 메트릭 로거.

    - 콘솔 출력 (실시간 모니터링)
    - CSV 파일 저장 (후처리/시각화용)
    - TensorBoard 지원 (선택)
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = False,
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # CSV 로거 설정
        self.csv_path = os.path.join(log_dir, "progress.csv")
        self.csv_file = None
        self.csv_writer = None
        self.csv_header_written = False
        self._all_rows: list = []           # 전체 행 누적 (헤더 확장 시 재작성용)
        self._all_fields: set = set()       # 지금까지 등장한 모든 키 추적

        # 메트릭 버퍼 (주기적 평균 계산용)
        self.metric_buffer: Dict[str, list] = defaultdict(list)

        # 에피소드 추적
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = time.time()

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard")
                self.tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                print("[Logger] TensorBoard를 사용할 수 없습니다. tensorboard 패키지를 설치하세요.")

    def log_scalar(self, key: str, value: float, step: Optional[int] = None):
        """단일 스칼라 메트릭을 기록합니다."""
        self.metric_buffer[key].append(value)
        if self.tb_writer and step is not None:
            self.tb_writer.add_scalar(key, value, step)

    def log_episode(self, episode_return: float, episode_length: int):
        """에피소드 결과를 기록합니다."""
        self.episode_count += 1
        self.log_scalar("episode/return", episode_return)
        self.log_scalar("episode/length", episode_length)

    def dump(self, step: int):
        """
        버퍼에 쌓인 메트릭의 평균을 계산하여
        콘솔과 CSV에 출력합니다.
        """
        self.total_steps = step
        elapsed = time.time() - self.start_time
        fps = step / elapsed if elapsed > 0 else 0

        # 평균 계산
        metrics = {}
        for key, values in self.metric_buffer.items():
            if values:
                metrics[key] = sum(values) / len(values)

        metrics["step"] = step
        metrics["time/elapsed_sec"] = elapsed
        metrics["time/fps"] = fps
        metrics["episode/count"] = self.episode_count

        # 콘솔 출력
        self._print_console(metrics)

        # CSV 출력
        self._write_csv(metrics)

        # 버퍼 초기화
        self.metric_buffer.clear()

    def _print_console(self, metrics: Dict[str, Any]):
        """메트릭을 콘솔에 출력합니다."""
        step = metrics.get("step", 0)
        elapsed = metrics.get("time/elapsed_sec", 0)
        fps = metrics.get("time/fps", 0)

        parts = [
            f"[Step {step:>8d}]",
            f"Time: {elapsed:.0f}s",
            f"FPS: {fps:.0f}",
        ]

        # 에피소드 관련 메트릭
        if "episode/return" in metrics:
            parts.append(f"Return: {metrics['episode/return']:.1f}")
        if "episode/length" in metrics:
            parts.append(f"EpLen: {metrics['episode/length']:.0f}")

        # 손실 관련 메트릭
        for key in sorted(metrics.keys()):
            if "loss" in key:
                parts.append(f"{key}: {metrics[key]:.4f}")

        print(" | ".join(parts))

    def _write_csv(self, metrics: Dict[str, Any]):
        """
        메트릭을 CSV 파일에 기록합니다.

        eval/* 등 새 키가 등장할 때마다 헤더를 확장하고 전체 파일을
        재작성합니다. 이전 행은 새 키 자리를 빈 문자열로 채웁니다.
        (이전 구현은 새 키 등장 시 헤더만 바꿔 기존 행을 모두 날렸습니다.)
        """
        # 이번 행을 누적
        self._all_rows.append(dict(metrics))
        new_keys = set(metrics.keys()) - self._all_fields
        self._all_fields |= set(metrics.keys())

        if not self.csv_header_written or new_keys:
            # 최초 작성 또는 새 키 등장 → 전체 파일 재작성
            if self.csv_file:
                self.csv_file.close()
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            fieldnames = sorted(self._all_fields)
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            for row in self._all_rows:
                self.csv_writer.writerow({k: row.get(k, "") for k in fieldnames})
            self.csv_header_written = True
        else:
            # 기존 헤더와 동일 → 마지막 행만 추가
            fieldnames = sorted(self._all_fields)
            self.csv_writer.writerow({k: metrics.get(k, "") for k in fieldnames})

        self.csv_file.flush()

    def close(self):
        """리소스를 정리합니다."""
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()
