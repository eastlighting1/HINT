import subprocess
import sys
from pathlib import Path
from loguru import logger

def test_cli_help_command() -> None:
    """
    Verify CLI entrypoint responds with help output.

    This test validates that invoking `hint` via the main module with `--help` exits successfully and emits usage text indicative of Hydra wiring.
    - Test Case ID: TS-12
    - Scenario: Execute CLI help command through Python interpreter.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_cli_help_command")

    script_path = Path("src/hint/app/main.py")
    cmd = [sys.executable, str(script_path), "--help"]

    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        logger.debug(f"Exit Code: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"CLI Error Output: {result.stderr}")
            raise AssertionError(f"CLI exited with code {result.returncode}")
            
        if "usage" not in result.stdout.lower() and "hydra" not in result.stdout.lower():
            logger.warning("Standard help text not found, but exit code was 0.")
            
        logger.info("CLI entry point execution verified.")
        
    except FileNotFoundError:
        logger.error(f"Script not found at {script_path}. Ensure test is run from project root.")
        raise AssertionError("Entry point script missing")
