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
    """
    def __init__(self, log_dir: Optional[Path] = None):
        # 1. 단일 Console 객체 생성 (이것을 핸들러와 프로그레스 바가 공유함)
        self.console = Console()
        self.metrics: Dict[str, list] = {}
        
        # Setup Logger
        self.logger = logging.getLogger("hint")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [] # Clear existing
        
        # Console Handler (Rich) - 깔끔한 출력을 위해 show_path=False
        ch = RichHandler(console=self.console, show_time=True, show_path=False, markup=True)
        self.logger.addHandler(ch)
        
        # File Handler
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "hint.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        transient=True: 완료 시 진행바가 사라져서 로그가 깔끔하게 유지됨.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True # <--- 핵심 수정: 완료된 바는 삭제
        )