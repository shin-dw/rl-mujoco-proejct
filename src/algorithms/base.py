"""
알고리즘 기본 클래스

모든 RL 알고리즘이 상속하는 추상 기반 클래스입니다.
통일된 인터페이스를 통해 학습/평가/저장/로드를 수행합니다.
"""

import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseAlgorithm(ABC):
    """
    강화학습 알고리즘 추상 기반 클래스.

    모든 알고리즘은 이 클래스를 상속하여 동일한 인터페이스를 보장합니다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: str = "auto",
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seed = seed

        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 랜덤 시드 설정
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # 학습 스텝 카운터
        self.total_steps = 0

    @abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        관측값으로부터 행동을 선택합니다.

        Args:
            obs: 현재 관측값
            deterministic: True면 탐색 없이 결정적 행동 반환

        Returns:
            선택된 행동 (numpy 배열)
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs) -> Dict[str, float]:
        """
        알고리즘 파라미터를 업데이트합니다.

        Returns:
            학습 메트릭 딕셔너리 (손실 값 등)
        """
        raise NotImplementedError

    def save(self, path: str):
        """모델을 저장합니다."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = self._get_save_state()
        state["total_steps"] = self.total_steps
        torch.save(state, path)
        print(f"[저장] 모델을 {path}에 저장했습니다. (Step: {self.total_steps})")

    def load(self, path: str):
        """저장된 모델을 불러옵니다."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.total_steps = state.get("total_steps", 0)
        self._load_save_state(state)
        print(f"[로드] 모델을 {path}에서 불러왔습니다. (Step: {self.total_steps})")

    @abstractmethod
    def _get_save_state(self) -> Dict[str, Any]:
        """저장할 상태 딕셔너리를 반환합니다."""
        raise NotImplementedError

    @abstractmethod
    def _load_save_state(self, state: Dict[str, Any]):
        """저장된 상태를 복원합니다."""
        raise NotImplementedError

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """NumPy 배열을 PyTorch 텐서로 변환합니다."""
        return torch.FloatTensor(data).to(self.device)
