from pathlib import Path
from unittest.mock import MagicMock
from hint.foundation.interfaces import TelemetryObserver, Registry

class TestFixtures:
    """
    Provides shared fixtures for unit and integration tests.
    Replaces pytest fixtures with explicit static methods.
    """

    @staticmethod
    def get_global_temp_dir(base_path: Path) -> Path:
        """
        Create and return a temporary directory for the test session.

        Args:
            base_path: The base directory to create the temp folder in.

        Returns:
            Path object to the created directory.
        """
        temp_dir = base_path / "hint_test_session"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    @staticmethod
    def get_mock_observer() -> TelemetryObserver:
        """
        Create a mock TelemetryObserver.

        Returns:
            MagicMock object mimicking TelemetryObserver.
        """
        obs = MagicMock(spec=TelemetryObserver)
        obs.create_progress.return_value.__enter__.return_value = MagicMock()
        return obs

    @staticmethod
    def get_mock_registry() -> Registry:
        """
        Create a mock Registry.

        Returns:
            MagicMock object mimicking Registry.
        """
        reg = MagicMock(spec=Registry)
        return reg