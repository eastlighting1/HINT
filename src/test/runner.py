import sys
import importlib
import inspect
import coverage
import hydra
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from omegaconf import DictConfig
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from jinja2 import Template

class TestRunner:
    """
    Custom test orchestration engine replacing pytest.
    Handles test discovery, execution, coverage measurement, and reporting.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.console = Console()
        self.cov = coverage.Coverage(source=["src/hint"], omit=["*/test/*", "*/site-packages/*"])
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Configure loguru and rich logging without interference.
        Loguru sinks to a file managed by Hydra, Rich handles terminal output.
        """
        logger.remove()
        
        # Rich Handler for Terminal
        logger.add(
            RichHandler(console=self.console, show_time=False, show_path=False),
            format="{message}",
            level="INFO"
        )

        # File Handler (Hydra output directory)
        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
            log_file = hydra_dir / "test_runner.log"
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                level="DEBUG",
                rotation="10 MB"
            )
            logger.info(f"Logging initialized. Log file: {log_file}")
        except Exception:
            logger.warning("Could not determine Hydra run directory. File logging disabled.")

    def discover_tests(self, search_dir: Path) -> List[Path]:
        """
        Recursively find python test files starting with 'test_'.

        Args:
            search_dir: Directory path to search.

        Returns:
            List of Path objects for found test files.
        """
        logger.info(f"Discovering tests in {search_dir}")
        return list(search_dir.rglob("test_*.py"))

    def run_tests(self, test_files: List[Path]) -> None:
        """
        Execute discovered tests and record results.

        Args:
            test_files: List of test file paths to execute.
        """
        self.cov.start()
        logger.info(f"Starting execution of {len(test_files)} test files...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Running Tests...", total=len(test_files))

            for test_file in test_files:
                progress.update(task, description=f"Processing {test_file.name}")
                self._run_single_module(test_file)
                progress.advance(task)

        self.cov.stop()
        self.cov.save()
        logger.info("Test execution completed.")

    def _run_single_module(self, file_path: Path) -> None:
        """
        Import a module and run functions starting with 'test_'.

        Args:
            file_path: Path to the python module.
        """
        module_name = str(file_path.relative_to(Path.cwd()).with_suffix("")).replace("/", ".")
        logger.debug(f"Importing module: {module_name}")

        try:
            mod = importlib.import_module(module_name)
            functions = inspect.getmembers(mod, inspect.isfunction)
            
            for name, func in functions:
                if name.startswith("test_"):
                    logger.debug(f"Running test case: {name}")
                    try:
                        func()
                        self.results.append({"file": file_path.name, "case": name, "status": "PASS", "error": None})
                    except Exception as e:
                        logger.error(f"Test Failed: {name} in {file_path.name} - {e}")
                        self.results.append({"file": file_path.name, "case": name, "status": "FAIL", "error": str(e)})

        except Exception as e:
            logger.critical(f"Failed to load module {file_path}: {e}")
            self.results.append({"file": file_path.name, "case": "Module Load", "status": "ERROR", "error": str(e)})

    def generate_report(self) -> None:
        """
        Generate HTML report using the provided template and coverage data.
        """
        logger.info("Generating report...")
        
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        report_filename = f"_{timestamp}.html"
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        report_path = artifacts_dir / report_filename

        # Load Template
        template_path = Path("resources/test_coverage_template.html")
        if not template_path.exists():
            logger.error(f"Template not found at {template_path}. Skipping report generation.")
            return

        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Mock coverage analysis for the template (In real scenario, parse self.cov data)
        # Using simplified data structure compatible with the template logic if identifiable
        # Assuming the template expects Jinja2 rendering based on standard practices
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASS'])
        failed_tests = total_tests - passed_tests
        
        render_data = {
            "title": "HINT Test Execution Report",
            "generated_at": timestamp,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "results": self.results,
            "coverage_score": round(self.cov.report(file=None), 2)  # Capture stdout coverage percent
        }

        try:
            # Basic replacement if template is simple, or Jinja2 if structure allows
            # Attempting Jinja2 rendering
            tm = Template(template_content)
            rendered_html = tm.render(**render_data)
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)
            
            logger.info(f"Report generated successfully: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to render report: {e}")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the test runner.

    Args:
        cfg: Hydra configuration object.
    """
    runner = TestRunner(cfg)
    
    # Determine scope
    root_test_dir = Path("src/test")
    target_dirs = [root_test_dir / "unit", root_test_dir / "integration", root_test_dir / "e2e"]
    
    all_test_files = []
    for d in target_dirs:
        if d.exists():
            all_test_files.extend(runner.discover_tests(d))
            
    runner.run_tests(all_test_files)
    runner.generate_report()
    
    # Check for failures and exit accordingly
    failures = [r for r in runner.results if r['status'] != 'PASS']
    if failures:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()