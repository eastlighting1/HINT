import sys
import ast
import importlib
import inspect
import coverage
import hydra
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from jinja2 import Environment, FileSystemLoader

class TestRunner:
    """
    Custom test orchestration engine replacing pytest.
    Handles test discovery, execution, coverage measurement, AST analysis, and report generation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.console = Console()
        self.cov = coverage.Coverage(
            source=["src/hint"], 
            omit=["*/test/*", "*/site-packages/*"],
            branch=True
        )
        self.results: List[Dict[str, Any]] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        logger.remove()
        
        logger.add(
            RichHandler(console=self.console, show_time=False, show_path=False),
            format="{message}",
            level=self.cfg.logging.level
        )

        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            log_file = hydra_dir / "test_runner.log"
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                level="DEBUG",
                rotation="10 MB"
            )
            logger.info(f"Logging initialized. Log file: {log_file}")
        except Exception:
            pass

    def discover_tests(self, search_dirs: List[Path]) -> List[Path]:
        test_files = []
        for d in search_dirs:
            if d.exists():
                logger.info(f"Discovering tests in {d}")
                test_files.extend(list(d.rglob("test_*.py")))
            else:
                logger.warning(f"Test directory not found: {d}")
        return sorted(test_files)

    def _get_module_name(self, file_path: Path) -> str:
        try:
            rel_path = file_path.relative_to(Path.cwd())
            return ".".join(rel_path.with_suffix("").parts)
        except ValueError:
            return file_path.stem

    def run_tests(self, test_files: List[Path]) -> bool:
        self.cov.start()
        logger.info(f"Starting execution of {len(test_files)} test files...")
        
        all_passed = True
        
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False 
        ) as progress:
            total_task = progress.add_task("[cyan]Running Test Suites...", total=len(test_files))

            for test_file in test_files:
                progress.update(total_task, description=f"Processing {test_file.name}")
                
                module_name = self._get_module_name(test_file)
                
                try:
                    if module_name in sys.modules:
                        mod = importlib.reload(sys.modules[module_name])
                    else:
                        mod = importlib.import_module(module_name)
                        
                except Exception as e:
                    all_passed = False
                    logger.critical(f"Failed to import module {module_name}: {e}")
                    self.results.append({
                        "file": test_file.name, "case": "Import", "status": "ERROR", 
                        "error": str(e), "traceback": traceback.format_exc()
                    })
                    progress.advance(total_task)
                    continue

                functions = inspect.getmembers(mod, inspect.isfunction)
                test_funcs = [(n, f) for n, f in functions if n.startswith("test_")]
                
                for name, func in test_funcs:
                    logger.debug(f"Running test case: {name}")
                    try:
                        func()
                        self.results.append({
                            "file": test_file.name, "case": name, "status": "PASS", "error": None
                        })
                    except AssertionError as e:
                        all_passed = False
                        logger.error(f"Assertion Failed: {name} - {e}")
                        self.results.append({
                            "file": test_file.name, "case": name, "status": "FAIL", 
                            "error": str(e), "traceback": traceback.format_exc()
                        })
                    except Exception as e:
                        all_passed = False
                        logger.error(f"Test Error: {name} - {e}")
                        self.results.append({
                            "file": test_file.name, "case": name, "status": "ERROR", 
                            "error": str(e), "traceback": traceback.format_exc()
                        })

                progress.advance(total_task)

        self.cov.stop()
        self.cov.save()
        logger.info("Test execution completed.")
        return all_passed

    def _calc_complexity(self, source_code: str) -> int:
        complexity = 1
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, 
                                     ast.With, ast.AsyncWith, ast.ExceptHandler, 
                                     ast.Assert, ast.comprehension)):
                    complexity += 1
                elif isinstance(node, (ast.BoolOp)):
                    complexity += len(node.values) - 1
        except Exception:
            pass 
        return complexity

    def _analyze_ast_structure(self, source_code: str) -> Tuple[List[Dict], List[Dict]]:
        classes = []
        methods = []
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append({"name": node.name, "line": node.lineno})
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = 1
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, 
                                              ast.With, ast.AsyncWith, ast.ExceptHandler, 
                                              ast.Assert, ast.comprehension)):
                            complexity += 1
                        elif isinstance(child, (ast.BoolOp)):
                            complexity += len(child.values) - 1
                    
                    methods.append({
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "complexity": complexity
                    })
        except Exception:
            pass
        return classes, methods

    def generate_report(self) -> None:
        logger.info("Generating quality report...")
        
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        report_dir = Path("artifacts")
        report_dir.mkdir(exist_ok=True, parents=True)
        report_path = report_dir / f"report_{timestamp}.html"

        files_list = []
        payload_json = {}
        total_missed_lines = 0
        risky_files_count = 0
        
        try:
            self.cov.load()
            measured_files = self.cov.get_data().measured_files()
            
            for file_path in measured_files:
                path_obj = Path(file_path)
                try:
                    display_name = str(path_obj.relative_to(Path.cwd()))
                except ValueError:
                    display_name = path_obj.name

                try:
                    analysis = self.cov.analysis2(file_path)
                    executable = analysis[1]
                    missing = analysis[3]
                    
                    n_stmts = len(executable)
                    n_miss = len(missing)
                    n_cover = n_stmts - n_miss
                    cov_pct = round((n_cover / n_stmts * 100), 1) if n_stmts > 0 else 100.0
                    
                    total_missed_lines += n_miss

                    with open(file_path, "r", encoding="utf-8") as f:
                        source_code = f.read()
                    
                    file_complexity = self._calc_complexity(source_code)
                    
                    is_risky = cov_pct < 50.0 and file_complexity >= 5
                    if is_risky:
                        risky_files_count += 1

                    files_list.append({
                        "name": display_name,
                        "statements": n_stmts,
                        "missed": n_miss,
                        "coverage": cov_pct,
                        "complexity": file_complexity
                    })

                    ast_classes, ast_methods = self._analyze_ast_structure(source_code)
                    
                    enriched_methods = []
                    for m in ast_methods:
                        m_lines = set(range(m["line"], m["end_line"] + 1))
                        m_exec = m_lines.intersection(executable)
                        m_miss = m_lines.intersection(missing)
                        
                        m_cov = 100.0
                        if m_exec:
                            m_cov = round(((len(m_exec) - len(m_miss)) / len(m_exec)) * 100, 1)
                        
                        status = "safe"
                        if m_cov < 50.0 and m["complexity"] >= 5:
                            status = "risk"
                        elif m["complexity"] >= 10:
                            status = "warning"
                            
                        m["coverage"] = m_cov
                        m["status"] = status
                        enriched_methods.append(m)

                    source_lines = []
                    for idx, line_content in enumerate(source_code.splitlines(), start=1):
                        hit_status = None
                        if idx in executable:
                            hit_status = idx not in missing
                        
                        source_lines.append({
                            "ln": idx,
                            "ct": line_content,
                            "hit": hit_status
                        })
                    
                    payload_json[display_name] = {
                        "source": source_lines,
                        "classes": ast_classes,
                        "methods": enriched_methods
                    }

                except Exception as e:
                    logger.warning(f"Could not analyze file {display_name}: {e}")
                    continue

            files_list.sort(key=lambda x: x['coverage'])

            try:
                global_cov = self.cov.report(file=None)
            except Exception:
                global_cov = 0.0
            
        except Exception as e:
            logger.error(f"Error during coverage analysis: {e}")
            global_cov = 0.0

        template_dir = Path("resources")
        if not (template_dir / "test_coverage_template.html").exists():
            template_dir = Path(".") 
            
        try:
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template("test_coverage_template.html")
            
            passed = len([r for r in self.results if r['status'] == 'PASS'])
            failed = len([r for r in self.results if r['status'] != 'PASS'])

            summary = {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "coverage": round(global_cov, 2),
                "risky_files": risky_files_count,
                "missed_lines": total_missed_lines,
                "total_files": len(files_list),
                "trend": 0
            }
            
            html_out = template.render(
                title="HINT Test Report",
                generated_at=timestamp,
                summary=summary,
                results=self.results,
                files=files_list,
                payload_json=payload_json
            )
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_out)
                
            logger.info(f"Report generated successfully: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            logger.debug(traceback.format_exc())

@hydra.main(version_base=None, config_path="../../configs", config_name="test_config")
def main(cfg: DictConfig) -> None:
    runner = TestRunner(cfg)
    project_root = Path.cwd()
    test_root = project_root / cfg.paths.test_root
    
    target_dirs = []
    if cfg.tests.run_unit: target_dirs.append(test_root / "unit")
    if cfg.tests.run_integration: target_dirs.append(test_root / "integration")
    if cfg.tests.run_e2e: target_dirs.append(test_root / "e2e")
    if not target_dirs: target_dirs = [test_root]

    test_files = runner.discover_tests(target_dirs)
    if not test_files:
        sys.exit(1)

    success = runner.run_tests(test_files)
    runner.generate_report()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()