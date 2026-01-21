"""Summary of the ventilation module.

Longer description of the module purpose and usage.
"""

import polars as pl

from pathlib import Path

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig



class VentilationTagger(PipelineComponent):

    """Summary of VentilationTagger purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """



    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.cfg = config

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

        resources_dir = Path(self.cfg.resources_dir)

        proc_dir = Path(self.cfg.proc_dir)



        vitals_path = proc_dir / self.cfg.artifacts.vitals_mean_file

        interventions_path = proc_dir / self.cfg.artifacts.interventions_file



        if not vitals_path.exists() or not interventions_path.exists():

            raise FileNotFoundError("Missing artifacts. Ensure TimeSeriesAggregator and OutcomesBuilder ran.")



        self.observer.log("INFO", "[1.4.1] Resolving ventilation labels")



        vent_itemids = {

            445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684,

            224685, 224684, 224686, 218, 436, 535, 444, 459, 224697, 224695, 224696,

            224746, 224747, 221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750,

            227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505, 506,

            686, 220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670,

            671, 672, 224701,

        }



        itemmap = pl.read_csv(resources_dir / "itemid_to_variable_map.csv").with_columns(

            pl.col("ITEMID").cast(pl.Int64)

        )



        vent_labels_df = (

            itemmap.filter(pl.col("ITEMID").is_in(list(vent_itemids)))

            .select(pl.col("MIMIC LABEL").alias("LABEL"))

            .unique()

        )



        vent_labels = vent_labels_df.to_series().to_list()

        self.observer.log("INFO", f"[1.4.1] Vent labels resolved. count={len(vent_labels)}")



        vl = pl.read_parquet(vitals_path)



        vent_events = (

            vl.filter(pl.col("LABEL").is_in(vent_labels))

            .select(["ICUSTAY_ID", "HOURS_IN"])

            .unique()

            .with_columns(pl.lit(1).cast(pl.Int8).alias("VENT_NEW"))

        )



        self.observer.log("INFO", f"[1.4.2] Vent events matched. rows={vent_events.height}")



        skeleton = pl.read_parquet(interventions_path)



        if "VENT" in skeleton.columns:

            skeleton = skeleton.drop("VENT")





        merged = (

            skeleton.join(vent_events, left_on=["ICUSTAY_ID", "HOUR_IN"], right_on=["ICUSTAY_ID", "HOURS_IN"], how="left")

            .sort(["ICUSTAY_ID", "HOUR_IN"])

            .with_columns(

                pl.col("VENT_NEW")

                .fill_null(strategy="forward", limit=6)

                .fill_null(0)

                .alias("VENT")

            )

            .drop("VENT_NEW")

        )



        merged.write_parquet(interventions_path)

        self.observer.log("INFO", f"[1.4.3] Vent flags updated. target={interventions_path.name}")
