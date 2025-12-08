from pathlib import Path
from typing import List

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig
from rich.console import Console


def _setup_logging(cfg: DictConfig) -> Console:
    """
    Configure Loguru sinks for file and console output without conflicting with Rich.

    Args:
        cfg: Hydra configuration for the test runner.

    Returns:
        Console instance bound to the logger.
    """
    logger.remove()
    console = Console()
    level = cfg.logging.level
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log_file = run_dir / "tests.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(lambda msg: console.print(msg, end=""), level=level, enqueue=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        enqueue=True,
    )
    logger.info("Logging configured for console and file outputs.")
    logger.debug("Log file path resolved to {}", log_file)
    return console


def _selected_test_paths(cfg: DictConfig) -> List[str]:
    """
    Build the list of test paths based on Hydra configuration flags.

    Args:
        cfg: Hydra configuration for the test runner.

    Returns:
        List of filesystem paths to pass into pytest.
    """
    project_root = Path(get_original_cwd())
    test_root = project_root / cfg.paths.test_root
    paths: List[Path] = []
    if cfg.tests.run_unit:
        paths.append(test_root / "unit")
    if cfg.tests.run_integration:
        paths.append(test_root / "integration")
    if cfg.tests.run_e2e:
        paths.append(test_root / "e2e")
    return [str(path) for path in paths if path.exists()]


def _build_pytest_args(cfg: DictConfig) -> List[str]:
    """
    Construct pytest arguments including coverage and reporting options.

    Args:
        cfg: Hydra configuration for the test runner.

    Returns:
        List of pytest CLI arguments.
    """
    args: List[str] = []
    args.extend(_selected_test_paths(cfg))
    for target in cfg.coverage.targets:
        args.append(f"--cov={target}")
    if cfg.coverage.branch:
        args.append("--cov-branch")
    config_path = cfg.coverage.config
    if config_path:
        args.append(f"--cov-config={config_path}")
    reports = cfg.coverage.reports
    if reports.term:
        args.append("--cov-report=term-missing")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if reports.xml:
        args.append(f"--cov-report=xml:{run_dir / reports.xml}")
    if reports.html:
        args.append(f"--cov-report=html:{run_dir / reports.html}")
    if reports.json:
        args.append(f"--cov-report=json:{run_dir / reports.json}")
    return args


@hydra.main(version_base=None, config_path="../../configs", config_name="test_config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint that configures logging and dispatches pytest with coverage.

    Args:
        cfg: Hydra configuration object loaded from test_config.yaml.
    """
    _setup_logging(cfg)
    pytest_args = _build_pytest_args(cfg)
    if not pytest_args:
        logger.warning("No test paths were selected; defaulting to src/test.")
        project_root = Path(get_original_cwd())
        pytest_args.append(str(project_root / "src" / "test"))
    logger.info("Starting pytest with arguments: {}", pytest_args)
    exit_code = pytest.main(pytest_args)
    if exit_code != 0:
        logger.error("Pytest reported failures with exit code {}", exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
