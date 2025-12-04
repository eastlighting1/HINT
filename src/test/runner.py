import sys
import subprocess
from pathlib import Path
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf

def build_pytest_command(cfg: DictConfig, project_root: Path) -> List[str]:
    """
    Construct the pytest command list based on the Hydra configuration.

    Args:
        cfg: The loaded Hydra configuration object.
        project_root: The absolute path to the project root directory.

    Returns:
        A list of strings representing the command to execute.
    """
    cmd = [sys.executable, "-m", "pytest"]
    
    # 1. Target Selection
    targets = cfg.test.get("targets", [])
    if not targets:
        targets = ["src/tests"]
    
    # Ensure targets are relative to project root or absolute
    for t in targets:
        cmd.append(str(t))

    # 2. Filtering Options
    keywords = cfg.test.get("keywords")
    if keywords:
        cmd.extend(["-k", keywords])
    
    markers = cfg.test.get("markers")
    if markers:
        cmd.extend(["-m", markers])

    # 3. Coverage Options
    cov_cfg = cfg.test.get("coverage", {})
    if cov_cfg.get("enabled", False):
        source_pkgs = cov_cfg.get("source_packages", ["src/hint"])
        output_dir = cov_cfg.get("output_dir", "coverage_report")
        report_types = cov_cfg.get("report_types", ["term-missing", "html"])

        # Map 'html' type to specific output path
        for pkg in source_pkgs:
            # Adjust package path to be relative to project root if needed
            pkg_path = project_root / pkg if not Path(pkg).is_absolute() else Path(pkg)
            cmd.append(f"--cov={pkg_path}")
        
        for r_type in report_types:
            if r_type == "html":
                report_path = project_root / output_dir
                cmd.append(f"--cov-report=html:{report_path}")
            else:
                cmd.append(f"--cov-report={r_type}")

    # 4. Verbosity and other flags
    cmd.append("-v")
    
    return cmd

@hydra.main(version_base=None, config_path="../../configs", config_name="test_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the test runner.

    Executes pytest in a subprocess using configurations defined in YAML.
    Avoids using argparse entirely.

    Args:
        cfg: Hydra configuration object.
    """
    # Resolve project root (assuming runner.py is in src/tests/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    
    print(f"[Runner] Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"[Runner] Project Root: {project_root}")

    command = build_pytest_command(cfg, project_root)
    print(f"[Runner] Executing command: {' '.join(command)}")

    try:
        # Run pytest from the project root to ensure imports work correctly
        result = subprocess.run(command, cwd=project_root)
        
        if result.returncode == 0:
            print("\n[Runner] Tests completed successfully.")
            if cfg.test.coverage.enabled:
                cov_dir = cfg.test.coverage.get("output_dir", "coverage_report")
                report_index = project_root / cov_dir / "index.html"
                print(f"[Runner] Coverage report available at: {report_index}")
        else:
            print(f"\n[Runner] Tests failed with exit code: {result.returncode}")
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n[Runner] Execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[Runner] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()