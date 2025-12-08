import polars as pl
from pathlib import Path
from typing import List

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class LabelGenerator(PipelineComponent):
    """
    Generates ONSET/WEAN/STAY ON/STAY OFF labels.
    Refactored to use vectorized Polars expressions instead of slow Python loops.
    """
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        self.observer.log("INFO", "LabelGenerator: Generating ventilation labels...")
        
        proc_dir = Path(self.cfg.proc_dir)
        ds_path = proc_dir / "dataset_123.parquet"
        ds = pl.read_parquet(ds_path)
        
        self.observer.log("INFO", "LabelGenerator: Filling sequences (vectorized)...")

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

        self.observer.log("INFO", "LabelGenerator: Calculating labels (vectorized)...")
        
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
        ).select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "VENT_CLASS"])

        min_offset = INPUT_WIN + GAP
        final_df = (
            ds.join(valid_labels, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], how="inner")
            .filter(pl.col("HOUR_IN") >= (pl.col("HOUR_IN").min().over("ICUSTAY_ID") + min_offset))
        )
        
        out_path = proc_dir / "dataset_123_answer.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(out_path)
        
        self.observer.log("INFO", f"LabelGenerator: Wrote dataset_123_answer.parquet (rows={final_df.height}) to {out_path}")
