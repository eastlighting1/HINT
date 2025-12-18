from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from hint.foundation.interfaces import Registry, TelemetryObserver

class BaseComponent(ABC):
    """
    모든 트레이닝 컴포넌트(Trainer, Evaluator)의 공통 부모 클래스.
    공통 인프라(Registry, Observer)와 디바이스 설정을 관리합니다.
    """
    def __init__(self, registry: Registry, observer: TelemetryObserver, device: str):
        self.registry = registry
        self.observer = observer
        self.device = device

class BaseTrainer(BaseComponent):
    """
    모델 학습 로직을 담당하는 추상 클래스.
    """
    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any, evaluator: 'BaseEvaluator', **kwargs) -> None:
        """
        학습 루프를 실행합니다.
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            evaluator: 검증을 수행할 Evaluator 인스턴스
        """
        pass

class BaseEvaluator(BaseComponent):
    """
    모델 평가 및 메트릭 계산을 담당하는 추상 클래스.
    """
    @abstractmethod
    def evaluate(self, loader: Any, **kwargs) -> Dict[str, float]:
        """
        평가를 수행하고 메트릭 딕셔너리를 반환합니다.
        """
        pass

class BaseDomainService(ABC):
    """
    데이터 준비, 컴포넌트 조립, 실행 흐름을 제어하는 오케스트레이터.
    """
    def __init__(self, observer: TelemetryObserver):
        self.observer = observer

    @abstractmethod
    def execute(self) -> None:
        """서비스의 메인 파이프라인을 실행합니다."""
        pass