import logging
from pathlib import Path
from typing import Optional, Dict

from hydra.core.hydra_config import HydraConfig
from loguru import logger as loguru_logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from ..foundation.interfaces import TelemetryObserver


class RichTelemetryObserver(TelemetryObserver):
    """Telemetry observer that logs to Rich and Hydra log files.

    Emits terminal logs through Rich while mirroring the same messages to the
    Hydra-managed log file via loguru without creating a new file.
    """

    def __init__(self) -> None:
        self.console = Console()
        self.metrics: Dict[str, list] = {}
        self.console_logger = self._build_console_logger()
        self.file_logger = self._build_file_logger()

    def _build_console_logger(self) -> logging.Logger:
        logger = logging.getLogger("hint.console")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            enable_link_path=False,
        )
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def _resolve_hydra_log_file(self) -> Optional[Path]:
        try:
            hydra_cfg = HydraConfig.get()
            output_dir = Path(hydra_cfg.runtime.output_dir)
            preferred = output_dir / "main.log"
            if preferred.exists():
                return preferred
            app_log = output_dir / f"{hydra_cfg.job.name}.log"
            if app_log.exists():
                return app_log
            existing = sorted(output_dir.glob("*.log"))
            if existing:
                return existing[0]
        except Exception:
            return None
        return None

    def _build_file_logger(self):
        loguru_logger.remove()
        log_path = self._resolve_hydra_log_file()
        if log_path is not None:
            loguru_logger.add(
                log_path,
                level="INFO",
                enqueue=True,
                format="[{time:YYYY-MM-DD HH:mm:ss}][{level}][{name}] {message}",
            )
        return loguru_logger.bind(channel="hint")

    def log(self, level: str, message: str) -> None:
        normalized = level.upper()
        self.console_logger.log(getattr(logging, normalized, logging.INFO), message)
        self.file_logger.log(normalized, message)

    def track_metric(self, name: str, value: float, step: int) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"step": step, "value": value})

    def create_progress(self, desc: str, total: int) -> Progress:
        """Create a Rich progress display for multi-step tasks.

        Args:
            desc (str): Description to show in the progress header.
            total (int): Number of steps in the task.

        Returns:
            Progress: Configured Rich progress instance.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
