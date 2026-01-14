"""Summary of the outcomes module.

Longer description of the module purpose and usage.
"""

import polars as pl

from pathlib import Path

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig, CNNConfig



class OutcomesBuilder(PipelineComponent):

    """Summary of OutcomesBuilder purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    cnn_cfg (Any): Description of cnn_cfg.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """



    def __init__(self, config: ETLConfig, cnn_config: CNNConfig, registry: Registry, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        cnn_config (Any): Description of cnn_config.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.cfg = config

        self.cnn_cfg = cnn_config

        self.registry = registry

        self.observer = observer



    def execute(self) -> None:

        """Summary of execute.
        
        Longer description of the execute behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        proc_dir = Path(self.cfg.proc_dir)

        patients_path = proc_dir / self.cfg.artifacts.patients_file

        if not patients_path.exists():

            raise FileNotFoundError(f"Dependency missing: {patients_path}. Run StaticExtractor first.")



        self.observer.log("INFO", "OutcomesBuilder: Stage 1/4 loading cohort")

        patients = pl.read_parquet(patients_path).select(

            ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STAY_HOURS"]

        )



        self.observer.log("INFO", "OutcomesBuilder: Stage 2/4 building hourly grid")

        max_hours = int(self.cnn_cfg.seq_len)

        skeleton = (

            patients

            .with_columns(

                pl.int_ranges(

                    0,

                    pl.min_horizontal(pl.col("STAY_HOURS"), pl.lit(max_hours))

                ).alias("HOUR_IN")

            )

            .explode("HOUR_IN")

            .drop("STAY_HOURS")

            .sort(["ICUSTAY_ID", "HOUR_IN"])

        )



        self.observer.log("INFO", f"OutcomesBuilder: Created {skeleton.height} hourly rows")



        self.observer.log("INFO", "OutcomesBuilder: Stage 3/4 initializing outcome flags")

        final_df = skeleton.with_columns([

            pl.lit(0).cast(pl.Int8).alias("OUTCOME_FLAG"),

            pl.lit(0).cast(pl.Int8).alias("VENT"),

            pl.lit(0).cast(pl.Int8).alias("VASO")

        ])



        self.observer.log("INFO", "OutcomesBuilder: Stage 4/4 saving outcome skeleton")

        out_path = proc_dir / self.cfg.artifacts.interventions_file

        final_df.write_parquet(out_path)

        self.observer.log("INFO", f"OutcomesBuilder: Saved event skeleton to {out_path}")
