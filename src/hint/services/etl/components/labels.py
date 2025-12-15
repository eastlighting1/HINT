import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class LabelGenerator(PipelineComponent):
    """
    Generate ONSET/WEAN/STAY ON/STAY OFF labels independently.

    Produces `labels.parquet` containing `ICUSTAY_ID`, `HOUR_IN`, and encoded `LABEL`.
    """
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        self.observer.log("INFO", "LabelGenerator: Generating independent ventilation labels...")
        
        proc_dir = Path(self.cfg.proc_dir)
        if not proc_dir.exists():
            proc_dir.mkdir(parents=True, exist_ok=True)

        ds_path = proc_dir / self.cfg.artifacts.features_file
        
        if not ds_path.exists():
             if not ds_path.is_absolute():
                 ds_path = Path.cwd() / ds_path
             if not ds_path.exists():
                 raise FileNotFoundError(f"Input dataset not found at {ds_path}")

        ds = pl.read_parquet(ds_path)
        
        self.observer.log("INFO", "LabelGenerator: Filling sequences...")

        bounds = (
            ds.group_by("ICUSTAY_ID")
            .agg([
                pl.min("HOUR_IN").alias("min_h"),
                pl.max("HOUR_IN").alias("max_h"),
                pl.first("SUBJECT_ID"),
                pl.first("HADM_ID")
            ])
        )
        
        grid_rows = []
        for row in bounds.iter_rows(named=True):
            h_start, h_end = row["min_h"], row["max_h"]
            sid, hid, iid = row["SUBJECT_ID"], row["HADM_ID"], row["ICUSTAY_ID"]
            for h in range(h_start, h_end + 1):
                grid_rows.append((sid, hid, iid, h))
                
        base = pl.DataFrame(
            grid_rows, 
            schema=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], 
            orient="row"
        )
        
        filled = (
            base.join(
                ds.select(["ICUSTAY_ID", "HOUR_IN", "VENT"]), 
                on=["ICUSTAY_ID", "HOUR_IN"], 
                how="left"
            )
            .sort(["ICUSTAY_ID", "HOUR_IN"])
            .with_columns(
                pl.col("VENT").fill_null(strategy="forward").fill_null(0).cast(pl.Int8)
            )
        )

        self.observer.log("INFO", "LabelGenerator: Calculating labels...")
        
        INPUT_WIN = self.cfg.input_window_h
        GAP = self.cfg.gap_h
        PRED_WIN = self.cfg.pred_window_h
        
        expressions = []
        for i in range(PRED_WIN):
            expressions.append(pl.col("VENT").shift(-i).alias(f"v_{i}"))
            
        windowed = filled.with_columns(expressions)

        cols = [pl.col(f"v_{i}") for i in range(PRED_WIN)]
        sum_expr = sum(cols)

        onset_cond = pl.lit(False)
        for i in range(PRED_WIN - 1):
            onset_cond = onset_cond | ((pl.col(f"v_{i+1}") - pl.col(f"v_{i}")) == 1)
            
        labels = (
            windowed
            .with_columns([
                sum_expr.alias("sum_v"),
                onset_cond.alias("has_onset")
            ])
            .with_columns(
                pl.when(pl.col("sum_v") == PRED_WIN).then(pl.lit("STAY ON"))
                .when(pl.col("sum_v") == 0).then(pl.lit("STAY OFF"))
                .when(pl.col("has_onset")).then(pl.lit("ONSET"))
                .otherwise(pl.lit("WEAN"))
                .alias("VENT_CLASS")
            )
        )

        valid_labels = labels.filter(
            pl.col(f"v_{PRED_WIN-1}").is_not_null()
        ).select(["ICUSTAY_ID", "HOUR_IN", "VENT_CLASS"])

        class_map = {"ONSET": 0, "WEAN": 1, "STAY ON": 2, "STAY OFF": 3}
        
        final_labels = (
            valid_labels
            .filter(pl.col("VENT_CLASS").is_in(class_map.keys()))
            .with_columns(
                pl.col("VENT_CLASS").replace(class_map).cast(pl.Int64).alias("LABEL")
            )
            .select(["ICUSTAY_ID", "HOUR_IN", "LABEL"])
        )
        
        out_path = proc_dir / self.cfg.artifacts.labels_file
        self.registry.save_labels(final_labels, out_path)
        
        self.observer.log("INFO", f"LabelGenerator: Saved labels to {out_path} (rows={final_labels.height})")
