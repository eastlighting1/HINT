import subprocess
import sys
from loguru import logger

def test_cli_help_command() -> None:
    """
    Validates that the application entry point is executable and responds to basic commands.
    
    Test Case ID: TS-12
    Description:
        Executes 'src/hint/app/main.py' as a subprocess with the '--help' argument.
        Verifies that the process exits with code 0.
        Checks if standard help text (e.g., Hydra info) is present in stdout.
    """
    logger.info("Starting test: test_cli_help_command")
    
    # Construct command: python src/hint/app/main.py --help
    # Note: Adjust path relative to where runner is executed (typically project root)
    script_path = "src/hint/app/main.py"
    
    cmd = [sys.executable, script_path, "--help"]
    
    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False # We check manually
        )
        
        logger.debug(f"Exit Code: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"CLI Error Output: {result.stderr}")
            raise AssertionError(f"CLI exited with code {result.returncode}")
            
        # Hydra help usually contains 'Powered by Hydra' or usage info
        if "usage" not in result.stdout.lower() and "hydra" not in result.stdout.lower():
            logger.warning("Standard help text not found, but exit code was 0.")
            
        logger.info("CLI entry point execution verified.")
        
    except FileNotFoundError:
        logger.error(f"Script not found at {script_path}. Ensure test is run from project root.")
        raise AssertionError("Entry point script missing")