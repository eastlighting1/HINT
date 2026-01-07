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
    """Telemetry observer with Rich console and Loguru file output.

    This implementation logs structured messages to the console and
    a Hydra-managed log file when available.

    Attributes:
        console (Console): Rich console instance.
        metrics (Dict[str, list]): Collected metrics keyed by name.
        console_logger (logging.Logger): Console logger with Rich handler.
        file_logger (Any): Loguru logger bound to the hint channel.
    """

    def __init__(self) -> None:
        """Initialize console and file loggers."""
        self.console = Console()
        self.metrics: Dict[str, list] = {}
        self.console_logger = self._build_console_logger()
        self.file_logger = self._build_file_logger()

    def _build_console_logger(self) -> logging.Logger:
        """Create a Rich-enabled console logger.

        Returns:
            logging.Logger: Configured console logger.
        """
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
        """Resolve the Hydra-managed log file path if present.

        Returns:
            Optional[Path]: Path to a log file when found, otherwise None.
        """
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
        """Create a Loguru logger bound to the hint channel.

        Returns:
            Any: Loguru logger instance bound with a channel field.
        """
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
        """Record a message to console and file loggers.

        Args:
            level (str): Logging level name.
            message (str): Log message content.
        """
        normalized = level.upper()
        self.console_logger.log(getattr(logging, normalized, logging.INFO), message)
        self.file_logger.log(normalized, message)

    def track_metric(self, name: str, value: float, step: int) -> None:
        """Store a metric value for later inspection.

        Args:
            name (str): Metric name.
            value (float): Metric value.
            step (int): Step index.
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"step": step, "value": value})

    def create_progress(self, desc: str, total: int) -> Progress:
        """Create a Rich progress bar for long-running tasks.

        Args:
            desc (str): Task description.
            total (int): Total progress count.

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
