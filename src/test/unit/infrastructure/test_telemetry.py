from unittest.mock import MagicMock, patch

from loguru import logger
from rich.console import Console
from rich.progress import Progress

from hint.infrastructure.telemetry import RichTelemetryObserver


def test_telemetry_initializes_console_and_loggers() -> None:
    """
    [One-line Summary] Verify RichTelemetryObserver builds console and logging sinks.

    [Description]
    Instantiate the telemetry observer with patched logger builders to confirm it constructs
    a Rich console and initializes console/file loggers without duplicating handlers.

    Test Case ID: INF-TEL-01
    Scenario: Instantiate telemetry observer with patched logger builders.

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
    [One-line Summary] Confirm RichTelemetryObserver stores tracked metrics with step metadata.

    [Description]
    Track sequential metric updates and assert the observer records step and value entries per
    key so downstream reporting remains deterministic.

    Test Case ID: INF-TEL-02
    Scenario: Track sequential metric updates for the same key.

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
    [One-line Summary] Validate progress creation uses the telemetry console.

    [Description]
    Request a progress instance from the telemetry observer and ensure it is bound to the
    observer console for consistent terminal rendering.

    Test Case ID: INF-TEL-03
    Scenario: Build a progress tracker from the telemetry observer.

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
