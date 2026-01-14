"""Summary of the main module.

Longer description of the module purpose and usage.
"""

import random
import sys

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

    logged_start = False



    if mode in ["all", "etl"]:

        etl_service = factory.create_etl_service()

        if not logged_start:

            etl_service.observer.log("INFO", f"App: Stage 1/3 initialize in mode={mode}.")

            logged_start = True

        etl_service.observer.log("INFO", "App: Stage 1/3 ETL pipeline start.")

        etl_service.run_pipeline()

        etl_service.observer.log("INFO", "App: Stage 1/3 ETL pipeline complete.")



    if mode in ["all", "icd"]:

        icd_service = factory.create_icd_service()

        if not logged_start:

            icd_service.observer.log("INFO", f"App: Stage 2/3 initialize in mode={mode}.")

            logged_start = True



        icd_service.observer.log("INFO", "App: Stage 2/3 ICD training start.")

        icd_service.execute()

        icd_service.observer.log("INFO", "App: Stage 2/3 ICD training complete.")



        icd_service.observer.log("INFO", "App: Stage 2/3 ICD feature injection start.")

        icd_service.generate_intervention_dataset(factory.ctx.cnn)

        icd_service.observer.log("INFO", "App: Stage 2/3 ICD feature injection complete.")



    if mode in ["all", "intervention"]:

        int_service = factory.create_intervention_service()

        if not logged_start:

            if int_service.observer:

                int_service.observer.log("INFO", f"App: Stage 3/3 initialize in mode={mode}.")

            logged_start = True



        if int_service.train_ds is not None:

            int_service.observer.log("INFO", "App: Stage 3/3 intervention training start.")

            int_service.execute()

            int_service.observer.log("INFO", "App: Stage 3/3 intervention training complete.")

        else:

            int_service.observer.log("ERROR", "App: Stage 3/3 intervention training skipped (no data).")



if __name__ == "__main__":

    main()
