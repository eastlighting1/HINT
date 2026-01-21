import logging
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


class InterceptHandler(logging.Handler):
    """Redirect standard logging records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_test_logging(
    log_file_path: Optional[Path] = None,
    level: str = "INFO",
    console: Optional[Console] = None,
) -> Console:
    """
    [One-line Summary] Configure Loguru and Rich logging for tests.

    [Description]
    Initialize a Rich-backed console handler and optional file sink aligned with
    Hydra-managed paths, and install an intercept handler so standard logging
    records are routed through Loguru.

    Test Case ID: TEST-INF-LOG-01
    Scenario: Initialize console and file logging once before test execution.

    Args:
        log_file_path: Optional path for Hydra-managed log file sink.
        level: Base log level applied to handlers.
        console: Optional Rich console instance to reuse across sinks.

    Returns:
        Console instance used by the configured handlers.
    """
    active_console = console or Console()
    logger.remove()

    if console is not None:
        logger.add(
            RichHandler(
                console=active_console,
                markup=True,
                rich_tracebacks=True,
                show_path=False,
            ),
            format="{message}",
            level=level.upper(),
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    if log_file_path is not None:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    return active_console
