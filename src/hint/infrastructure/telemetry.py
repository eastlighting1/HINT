"""Summary of the telemetry module.

Longer description of the module purpose and usage.
"""

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

    """Summary of RichTelemetryObserver purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    _active_progress (Any): Description of _active_progress.
    console (Any): Description of console.
    console_logger (Any): Description of console_logger.
    file_logger (Any): Description of file_logger.
    metrics (Any): Description of metrics.
    """



    def __init__(self) -> None:

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.console = Console()

        self.metrics: Dict[str, list] = {}

        self.console_logger = self._build_console_logger()

        self.file_logger = self._build_file_logger()

        self._active_progress: Optional[Progress] = None



    def _build_console_logger(self) -> logging.Logger:

        """Summary of _build_console_logger.
        
        Longer description of the _build_console_logger behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        logging.Logger: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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

        """Summary of _resolve_hydra_log_file.
        
        Longer description of the _resolve_hydra_log_file behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Optional[Path]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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

        """Summary of _build_file_logger.
        
        Longer description of the _build_file_logger behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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

        """Summary of log.
        
        Longer description of the log behavior and usage.
        
        Args:
        level (Any): Description of level.
        message (Any): Description of message.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        normalized = level.upper()

        self.console_logger.log(getattr(logging, normalized, logging.INFO), message)

        self.file_logger.log(normalized, message)



    def track_metric(self, name: str, value: float, step: int) -> None:

        """Summary of track_metric.
        
        Longer description of the track_metric behavior and usage.
        
        Args:
        name (Any): Description of name.
        value (Any): Description of value.
        step (Any): Description of step.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if name not in self.metrics:

            self.metrics[name] = []

        self.metrics[name].append({"step": step, "value": value})



    def create_progress(self, desc: str, total: int) -> Progress:

        """Summary of create_progress.
        
        Longer description of the create_progress behavior and usage.
        
        Args:
        desc (Any): Description of desc.
        total (Any): Description of total.
        
        Returns:
        Progress: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        progress = Progress(

            SpinnerColumn(),

            TextColumn("[bold blue]{task.description}"),

            BarColumn(bar_width=40),

            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),

            TimeRemainingColumn(),

            console=self.console,

            transient=True,

        )

        self._active_progress = progress

        return progress



    def get_child_progress(self) -> Optional[Progress]:

        """Summary of get_child_progress.
        
        Longer description of the get_child_progress behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Optional[Progress]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return self._active_progress
