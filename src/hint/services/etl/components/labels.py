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
        
        # --- 1. Sequence Filling (Vectorized) ---
        self.observer.log("INFO", "LabelGenerator: Filling sequences (vectorized)...")
        
        # Calculate min/max hours per stay
        bounds = (
            ds.group_by("ICUSTAY_ID")
            .agg([
                pl.min("HOUR_IN").alias("min_h"),
                pl.max("HOUR_IN").alias("max_h"),
                pl.first("SUBJECT_ID"),
                pl.first("HADM_ID")
            ])
        )
        
        # Create full grid
        # Note: This list comprehension is unavoidable for range generation but fast enough for metadata
        grid_rows = []
        for row in bounds.iter_rows(named=True):
            h_start, h_end = row["min_h"], row["max_h"]
            sid, hid, iid = row["SUBJECT_ID"], row["HADM_ID"], row["ICUSTAY_ID"]
            # Generate range [min_h, max_h] inclusive
            for h in range(h_start, h_end + 1):
                grid_rows.append((sid, hid, iid, h))
                
        base = pl.DataFrame(
            grid_rows, 
            schema=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], 
            orient="row"
        )
        
        # Join and forward fill
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

        # --- 2. Window Labeling (Vectorized) ---
        self.observer.log("INFO", "LabelGenerator: Calculating labels (vectorized)...")
        
        INPUT_WIN = self.cfg.input_window_h
        GAP = self.cfg.gap_h
        PRED_WIN = self.cfg.pred_window_h
        TOTAL_WIN = INPUT_WIN + GAP + PRED_WIN
        
        # We need to look ahead to determine label
        # Window logic: at time t (start of prediction window), look at [t, t+PRED_WIN)
        # But original logic anchors at t_0 and looks at t_0 + INPUT + GAP.
        # Let's align with original logic:
        # Original: labels.append(...) for window starting at index s + INPUT + GAP
        # Meaning: Label is assigned to the timestamp WHERE THE PREDICTION STARTS.
        
        # Shift VENT column to create future views
        # We want to check VENT status in [t, t+PRED_WIN)
        # Shift -1 means "next hour"
        
        expressions = []
        for i in range(PRED_WIN):
            expressions.append(pl.col("VENT").shift(-i).alias(f"v_{i}"))
            
        windowed = filled.with_columns(expressions)
        
        # Define conditions
        # STAY ON: All 1s
        # STAY OFF: All 0s
        # ONSET: 0 -> 1 transition
        # WEAN: 1 -> 0 transition
        
        cols = [pl.col(f"v_{i}") for i in range(PRED_WIN)]
        sum_expr = sum(cols)
        
        # Transition check: is there any (v_{i+1} - v_{i}) == 1?
        # ONSET condition: any increase
        onset_cond = pl.lit(False)
        for i in range(PRED_WIN - 1):
            onset_cond = onset_cond | ((pl.col(f"v_{i+1}") - pl.col(f"v_{i}")) == 1)
            
        # WEAN condition: any decrease
        # (Original code: ONSET if 1 in delta else WEAN) - implies WEAN is default if mixed?
        # Original: labels.append("ONSET" if 1 in delta else "WEAN")
        # So if ANY onset occurs, it's ONSET. If not, and not stay on/off, it's WEAN.
        
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
        
        # Filter invalid windows (end of stay)
        # The shift operation introduces nulls at the end of each group
        # We also need to respect the Input+Gap lag.
        # The label computed at 'filled' row T corresponds to the prediction starting at T.
        # But the original code aligns this label to T = t_start + INPUT + GAP.
        # So we need to shift the calculated label BACK by (INPUT + GAP) to align?
        # NO, original code: label_hours = [h0 + (s + INPUT + GAP) ...]
        # This means at hour H, we predict the window starting at H.
        # So the label computed above at row T is valid for hour T.
        # We just need to remove the last (PRED_WIN - 1) rows per stay where windows are incomplete.
        
        valid_labels = labels.filter(
            pl.col(f"v_{PRED_WIN-1}").is_not_null()
        ).select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "VENT_CLASS"])
        
        # Join back to features
        # Features are at HOUR_IN. We want to predict window [HOUR_IN, HOUR_IN + PRED_WIN]
        # But we need input data from [HOUR_IN - GAP - INPUT, HOUR_IN - GAP]
        # Wait, the dataset_123.parquet contains features at HOUR_IN.
        # If we want to predict 'VENT_CLASS' for the window starting at HOUR_IN,
        # we attach the label to the row at HOUR_IN.
        
        # However, the original logic was:
        # label_hours = h0 + s + INPUT + GAP
        # This implies we only generate labels for hours where we have enough history?
        # Yes, standard supervised learning.
        # So we need to ensure we only keep rows where HOUR_IN >= min_h + INPUT + GAP.
        
        min_offset = INPUT_WIN + GAP
        final_df = (
            ds.join(valid_labels, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], how="inner")
            .filter(pl.col("HOUR_IN") >= (pl.col("HOUR_IN").min().over("ICUSTAY_ID") + min_offset))
        )
        
        out_path = proc_dir / "dataset_123_answer.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(out_path)
        
        self.observer.log("INFO", f"LabelGenerator: Wrote dataset_123_answer.parquet (rows={final_df.height}) to {out_path}")