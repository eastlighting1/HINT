import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig


class VentilationTagger(PipelineComponent):
    """Annotate ventilation status on intervention events.

    Reads charted ventilation signals, resolves labels, and flags matching hours in the interventions table.

    Attributes:
        cfg (ETLConfig): Pipeline configuration.
        registry (Registry): Artifact registry for persistence.
        observer (TelemetryObserver): Telemetry adapter for rich and file logging.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Derive ventilation flags and update the interventions parquet.

        Loads vital and intervention artifacts, maps ventilation item identifiers to labels, and writes the updated file in place.

        Returns:
            None

        Raises:
            FileNotFoundError: If required parquet resources are missing.
        """
        resources_dir = Path(self.cfg.resources_dir)
        proc_dir = Path(self.cfg.proc_dir)
        self.observer.log("INFO", f"VentilationTagger: Resolving resources at {resources_dir} and {proc_dir}")

        vent_itemids = {
            445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684,
            224685, 224684, 224686, 218, 436, 535, 444, 459, 224697, 224695, 224696,
            224746, 224747, 221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750,
            227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505, 506,
            686, 220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670,
            671, 672, 224701,
        }

        itemmap = pl.read_csv(resources_dir / "itemid_to_variable_map.csv").with_columns(
            [
                pl.col("ITEMID").cast(pl.Int64),
                pl.col("MIMIC LABEL").alias("LABEL"),
            ]
        )
        self.observer.log("INFO", "VentilationTagger: Loaded item map for ventilation label lookup")

        vent_labels = (
            itemmap.filter(pl.col("ITEMID").is_in(list(vent_itemids)) & (pl.col("LINKSTO") == "chartevents"))
            .select("LABEL")
            .unique()
            .to_series()
            .to_list()
        )
        self.observer.log("INFO", f"VentilationTagger: Derived {len(vent_labels)} ventilation labels from item map")

        vl = pl.read_parquet(proc_dir / "vitals_labs.parquet")
        self.observer.log("INFO", f"VentilationTagger: Loaded vitals_labs.parquet with {vl.height} rows")

        vent_times = (
            vl.filter(pl.col("LABEL").is_in(vent_labels))
            .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOURS_IN"])
            .unique()
            .with_columns(pl.lit(1).alias("VENT"))
            .rename({"HOURS_IN": "HOUR_IN"})
        )
        self.observer.log("INFO", f"VentilationTagger: Matched ventilation windows count={vent_times.height}")

        iv = pl.read_parquet(proc_dir / "interventions.parquet")
        self.observer.log("INFO", f"VentilationTagger: Loaded interventions.parquet with {iv.height} rows")

        iv_out = (
            iv.join(vent_times, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], how="left")
            .with_columns(pl.col("VENT").fill_null(0).cast(pl.Int8))
            .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "OUTCOME_FLAG", "VENT"])
            .sort(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"])
        )

        iv_out.write_parquet(proc_dir / "interventions.parquet")
        self.observer.log("INFO", f"VentilationTagger: Updated interventions.parquet rows={iv_out.height}")
