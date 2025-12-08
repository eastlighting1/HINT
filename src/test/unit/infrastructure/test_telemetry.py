from unittest.mock import MagicMock, patch

from loguru import logger
from rich.console import Console
from rich.progress import Progress

from hint.infrastructure.telemetry import RichTelemetryObserver


def test_telemetry_initializes_console_and_loggers() -> None:
    """
    Verify RichTelemetryObserver builds console and logging sinks.

    This test validates that telemetry initialization constructs a Rich console and invokes console and file logger builders to avoid handler conflicts.
    - Test Case ID: INF-TEL-01
    - Scenario: Instantiate telemetry observer with patched logger builders.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_telemetry_initializes_console_and_loggers")

    with patch.object(RichTelemetryObserver, "_build_console_logger", return_value=MagicMock()) as console_builder, \
         patch.object(RichTelemetryObserver, "_build_file_logger", return_value=MagicMock()) as file_builder:
        observer = RichTelemetryObserver()

    assert isinstance(observer.console, Console)
    console_builder.assert_called_once()
    file_builder.assert_called_once()

    logger.info("Telemetry initialization verified.")


def test_telemetry_tracks_metrics() -> None:
    """
    Confirm RichTelemetryObserver stores tracked metrics with step metadata.

    This test validates that calling `track_metric` appends values keyed by metric name, preserving both value and step for deterministic reporting.
    - Test Case ID: INF-TEL-02
    - Scenario: Track sequential metric updates for the same key.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_telemetry_tracks_metrics")

    with patch.object(RichTelemetryObserver, "_build_console_logger", return_value=MagicMock()), \
         patch.object(RichTelemetryObserver, "_build_file_logger", return_value=MagicMock()):
        observer = RichTelemetryObserver()

    observer.track_metric("loss", 0.5, step=1)
    observer.track_metric("loss", 0.4, step=2)

    assert observer.metrics["loss"][0] == {"step": 1, "value": 0.5}
    assert observer.metrics["loss"][1] == {"step": 2, "value": 0.4}

    logger.info("Telemetry metric tracking verified.")


def test_telemetry_creates_progress_with_console() -> None:
    """
    Validate progress creation uses the telemetry console.

    This test ensures `create_progress` returns a Rich Progress instance bound to the observer console for consistent terminal rendering.
    - Test Case ID: INF-TEL-03
    - Scenario: Build a progress tracker from the telemetry observer.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_telemetry_creates_progress_with_console")

    with patch.object(RichTelemetryObserver, "_build_console_logger", return_value=MagicMock()), \
         patch.object(RichTelemetryObserver, "_build_file_logger", return_value=MagicMock()):
        observer = RichTelemetryObserver()

    progress = observer.create_progress(desc="demo", total=2)

    assert isinstance(progress, Progress)
    assert progress.console is observer.console

    logger.info("Telemetry progress creation verified.")
