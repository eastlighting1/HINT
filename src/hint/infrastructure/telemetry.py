import logging
from pathlib import Path
from typing import Any, Optional, Dict
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from ..foundation.interfaces import TelemetryObserver

class RichTelemetryObserver(TelemetryObserver):
    """
    Telemetry observer using Rich for console output and standard logging for files.
    Fixed: Prevent duplicate logs by disabling propagation.
    """
    def __init__(self, log_dir: Optional[Path] = None):
        # 1. 단일 Console 객체 생성
        self.console = Console()
        self.metrics: Dict[str, list] = {}
        
        # Setup Logger
        self.logger = logging.getLogger("hint")
        self.logger.setLevel(logging.INFO)
        
        # [Critical Fix] 중복 출력 방지: 상위 로거(Hydra/Root)로 전파 금지
        self.logger.propagate = False
        
        # 기존 핸들러 제거 (중복 방지)
        self.logger.handlers = []
        
        # 2. Console Handler (Rich) - 터미널용 깔끔한 출력
        # markup=True: 로그 메시지의 색상 코드 해석
        # enable_link_path=False: 경로 링크 비활성화 (지저분함 방지)
        ch = RichHandler(
            console=self.console, 
            show_time=True, 
            show_path=False, 
            markup=True,
            enable_link_path=False
        )
        self.logger.addHandler(ch)
        
        # 3. File Handler - 파일용 상세 출력 (타임스탬프 포함)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "hint.log", encoding='utf-8')
            # 파일에는 날짜/시간/레벨 등 상세 정보 기록
            formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log(self, level: str, message: str) -> None:
        lvl = level.upper()
        if lvl == "INFO":
            self.logger.info(message)
        elif lvl == "WARNING":
            self.logger.warning(message)
        elif lvl == "ERROR":
            self.logger.error(message)
        elif lvl == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(f"[{lvl}] {message}")

    def track_metric(self, name: str, value: float, step: int) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"step": step, "value": value})

    def create_progress(self, desc: str, total: int) -> Progress:
        """
        Returns a configured Rich Progress object.
        transient=True: 완료 시 사라짐 (로그 가독성 확보)
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True
        )