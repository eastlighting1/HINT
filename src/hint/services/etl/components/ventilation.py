import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class VentilationTagger(PipelineComponent):
    """Annotate ventilation status on intervention events.
    
    Reads extracted Vitals/Labs (Chartevents), identifies ventilation-related items,
    and updates the dense intervention skeleton.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        resources_dir = Path(self.cfg.resources_dir)
        proc_dir = Path(self.cfg.proc_dir)
        
        vitals_path = proc_dir / self.cfg.artifacts.vitals_mean_file
        interventions_path = proc_dir / self.cfg.artifacts.interventions_file

        if not vitals_path.exists() or not interventions_path.exists():
             raise FileNotFoundError(f"Missing artifacts. Ensure TimeSeriesAggregator and OutcomesBuilder ran.")

        self.observer.log("INFO", "VentilationTagger: Stage 1/3 Resolving ventilation labels")

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
        self.observer.log("INFO", f"VentilationTagger: Found {len(vent_labels)} labels for ventilation.")

        vl = pl.read_parquet(vitals_path)
        
        vent_events = (
            vl.filter(pl.col("LABEL").is_in(vent_labels))
            .select(["ICUSTAY_ID", "HOURS_IN"])
            .unique()
            .with_columns(pl.lit(1).cast(pl.Int8).alias("VENT_NEW"))
        )
        
        self.observer.log("INFO", f"VentilationTagger: Stage 2/3 Matched {vent_events.height} ventilation hours.")

        skeleton = pl.read_parquet(interventions_path)
        
        if "VENT" in skeleton.columns:
            skeleton = skeleton.drop("VENT")

        merged = (
            skeleton.join(vent_events, left_on=["ICUSTAY_ID", "HOUR_IN"], right_on=["ICUSTAY_ID", "HOURS_IN"], how="left")
            .with_columns(pl.col("VENT_NEW").fill_null(0).alias("VENT"))
            .drop("VENT_NEW")
        )

        merged.write_parquet(interventions_path)
        self.observer.log("INFO", f"VentilationTagger: Updated {interventions_path.name} with ventilation flags.")