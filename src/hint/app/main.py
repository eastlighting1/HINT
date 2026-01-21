"""Summary of the main module.

Longer description of the module purpose and usage.
"""

import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

from hint.app.factory import AppFactory



def _seed_everything(seed: int) -> None:
    """Summary of _seed_everything.
    
    Longer description of the _seed_everything behavior and usage.
    
    Args:
    seed (Any): Description of seed.
    
    Returns:
    None: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main() -> None:

    """Summary of main.
    
    Longer description of the main behavior and usage.
    
    Args:
    None (None): This function does not accept arguments.
    
    Returns:
    None: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
    """

    factory = AppFactory()

    _seed_everything(factory.ctx.seed)

    mode = factory.ctx.mode

    start_time = time.time()

    status = "success"

    error_detail = None

    stage0_observer = factory.create_telemetry()

    def _resolve_git_commit() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    stage0_observer.log("INFO", "[STAGE 0 START] Pipeline initialization.")

    if hasattr(stage0_observer, "render_stage0_receipt"):

        stage0_observer.render_stage0_receipt(factory.ctx)

    stage0_observer.log("INFO", f"[0.1] Configuration loaded. mode={mode}")

    stage0_observer.log("INFO", f"[0.2] Global RNG seeded. seed={factory.ctx.seed}")

    stage0_observer.log("INFO", f"[0.3] Run directory ready. run_dir={factory.run_dir}")

    stage0_observer.log("INFO", "[STAGE 0 END] System ready.")

    icd_best_model = None
    icd_best_metric = None
    intervention_best_metric = None

    try:

        if mode in ["all", "etl"]:

            stage_start = time.time()

            etl_service = factory.create_etl_service()

            etl_service.observer.log("INFO", "[STAGE 1 START] ETL pipeline start.")

            etl_service.run_pipeline()

            etl_service.observer.log("INFO", "[STAGE 1 END] ETL pipeline complete.")

            duration = time.time() - stage_start
            if hasattr(etl_service.observer, "trace_event"):
                etl_service.observer.trace_event("stage_end", {"stage": "etl", "duration_sec": duration})
            if hasattr(etl_service.observer, "trace_bottleneck"):
                etl_service.observer.trace_bottleneck("etl", duration, threshold_sec=300.0)


        if mode in ["all", "icd"]:

            stage_start = time.time()

            icd_service = factory.create_icd_service()

            icd_service.observer.log("INFO", "[STAGE 2 START] ICD training start.")

            icd_service.execute()

            icd_service.observer.log("INFO", "[STAGE 2 END] ICD training complete.")

            icd_best_model = getattr(icd_service, "best_model_name", None)
            icd_best_metric = getattr(getattr(icd_service, "entity", None), "best_metric", None)

            duration = time.time() - stage_start
            if hasattr(icd_service.observer, "trace_event"):
                icd_service.observer.trace_event("stage_end", {"stage": "icd", "duration_sec": duration})
            if hasattr(icd_service.observer, "trace_bottleneck"):
                icd_service.observer.trace_bottleneck("icd", duration, threshold_sec=300.0)


            stage_start = time.time()

            icd_service.observer.log("INFO", "[STAGE 2B START] Feature injection start.")

            icd_service.generate_intervention_dataset(factory.ctx.cnn)

            icd_service.observer.log("INFO", "[STAGE 2B END] Feature injection complete.")

            duration = time.time() - stage_start
            if hasattr(icd_service.observer, "trace_event"):
                icd_service.observer.trace_event("stage_end", {"stage": "feature_injection", "duration_sec": duration})
            if hasattr(icd_service.observer, "trace_bottleneck"):
                icd_service.observer.trace_bottleneck("feature_injection", duration, threshold_sec=300.0)


        if mode in ["all", "intervention"]:

            stage_start = time.time()

            int_service = factory.create_intervention_service()

            if int_service.train_ds is not None:

                int_service.observer.log("INFO", "[STAGE 3 START] Intervention training start.")

                int_service.execute()

                int_service.observer.log("INFO", "[STAGE 3 END] Intervention training complete.")

                intervention_best_metric = getattr(getattr(int_service, "entity", None), "best_metric", None)

            else:

                int_service.observer.log("ERROR", "[STAGE 3 SKIP] Intervention training skipped (no data).")

            duration = time.time() - stage_start
            if hasattr(int_service.observer, "trace_event"):
                int_service.observer.trace_event("stage_end", {"stage": "intervention", "duration_sec": duration})
            if hasattr(int_service.observer, "trace_bottleneck"):
                int_service.observer.trace_bottleneck("intervention", duration, threshold_sec=300.0)

    except Exception as exc:

        status = "error"

        error_detail = f"{exc}"

        stage0_observer.log("CRITICAL", f"[RUN END] status=error error={exc}")

        stage0_observer.log("ERROR", traceback.format_exc())

        raise

    finally:

        end_time = time.time()

        summary = {
            "status": status,
            "mode": mode,
            "seed": factory.ctx.seed,
            "run_dir": str(factory.run_dir),
            "git_commit": _resolve_git_commit(),
            "start_time": start_time,
            "end_time": end_time,
            "total_time_sec": end_time - start_time,
            "error": error_detail,
            "icd_best_model": icd_best_model,
            "icd_best_metric": icd_best_metric,
            "intervention_best_metric": intervention_best_metric,
        }

        metrics_dir = Path(factory.run_dir) / "metrics"

        metrics_dir.mkdir(parents=True, exist_ok=True)

        summary_path = metrics_dir / "summary.json"

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if status == "success":

            stage0_observer.log("SUCCESS", "[RUN END] status=success")

        if hasattr(stage0_observer, "render_run_end"):

            message = "Pipeline finished successfully." if status == "success" else "Pipeline terminated with errors."

            stage0_observer.render_run_end(status, message)



if __name__ == "__main__":

    main()
