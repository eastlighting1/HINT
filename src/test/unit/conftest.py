from hint.domain.vo import ETLConfig, CNNConfig

class UnitFixtures:
    """
    Provides specific fixtures for unit tests in the domain and foundation layers.
    This class replaces pytest fixtures, allowing explicit instantiation of test objects.
    """

    @staticmethod
    def get_minimal_etl_config() -> ETLConfig:
        """
        Creates a minimal ETLConfig object for testing purposes.

        Returns:
            ETLConfig: A valid configuration object with temporary paths.
        """
        return ETLConfig(
            raw_dir="/tmp/raw",
            proc_dir="/tmp/proc",
            resources_dir="/tmp/res"
        )

    @staticmethod
    def get_minimal_cnn_config() -> CNNConfig:
        """
        Creates a minimal CNNConfig object with default settings.

        Returns:
            CNNConfig: A valid CNN configuration object.
        """
        return CNNConfig(
            data_path="/tmp/data.h5",
            data_cache_dir="/tmp/cache",
            batch_size=16,
            epochs=1
        )