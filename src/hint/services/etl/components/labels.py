import polars as pl
from typing import List

from hint.foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from hint.domain.vo import ETLConfig

class LabelGenerator(PipelineComponent):
    """
    Generates ONSET/WEAN/STAY ON/STAY OFF labels.
    Ported from make_data.py: step_build_vent_labels
    """
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        self.observer.log("INFO", "LabelGenerator: Generating ventilation labels...")
        ds = self.registry.load_dataframe("dataset_123.parquet")
        
        # 1. Fill Sequence Logic
        def fill_sequence_for_vent(df: pl.DataFrame) -> pl.DataFrame:
            df = df.sort("HOUR_IN")
            hmin, hmax = int(df["HOUR_IN"].min()), int(df["HOUR_IN"].max())
            base = pl.DataFrame({"HOUR_IN": pl.arange(hmin, hmax + 1, eager=True, dtype=pl.Int32)})
            subj = int(df["SUBJECT_ID"][0])
            hadm = int(df["HADM_ID"][0])
            stay = int(df["ICUSTAY_ID"][0])
            
            return (
                base.join(df.select(["HOUR_IN", "VENT"]), on="HOUR_IN", how="left")
                .with_columns(pl.col("VENT").cast(pl.Int8).fill_null(strategy="forward").fill_null(0))
                .with_columns(
                    SUBJECT_ID=pl.lit(subj, dtype=pl.Int64),
                    HADM_ID=pl.lit(hadm, dtype=pl.Int64),
                    ICUSTAY_ID=pl.lit(stay, dtype=pl.Int64),
                )
            )

        # 2. Window Labeling Logic
        TOTAL_WIN = self.cfg.input_window_h + self.cfg.gap_h + self.cfg.pred_window_h
        
        def label_windows(vents: List[int]) -> List[str]:
            labels = []
            n = len(vents)
            if n < TOTAL_WIN: return labels
            for s in range(0, n - TOTAL_WIN + 1):
                pw = vents[s + self.cfg.input_window_h + self.cfg.gap_h : s + TOTAL_WIN]
                if all(v == 1 for v in pw): labels.append("STAY ON")
                elif all(v == 0 for v in pw): labels.append("STAY OFF")
                else:
                    delta = [pw[i+1] - pw[i] for i in range(len(pw)-1)]
                    labels.append("ONSET" if 1 in delta else "WEAN")
            return labels

        def per_stay_labels(df: pl.DataFrame) -> pl.DataFrame:
            vents = df["VENT"].to_list()
            labels = label_windows(vents)
            if not labels:
                return pl.DataFrame(schema={"SUBJECT_ID": pl.Int64, "HADM_ID": pl.Int64, "ICUSTAY_ID": pl.Int64, "HOUR_IN": pl.Int32, "VENT_CLASS": pl.Utf8})
            
            h0 = int(df["HOUR_IN"][0])
            label_hours = [h0 + (s + self.cfg.input_window_h + self.cfg.gap_h) for s in range(len(labels))]
            
            return pl.DataFrame({
                "SUBJECT_ID": [df["SUBJECT_ID"][0]] * len(labels),
                "HADM_ID": [df["HADM_ID"][0]] * len(labels),
                "ICUSTAY_ID": [df["ICUSTAY_ID"][0]] * len(labels),
                "HOUR_IN": label_hours,
                "VENT_CLASS": labels
            })

        seq_df = ds.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "VENT"])
        
        # Apply grouping and mapping
        # Note: Polars map_groups is used. This might be slow in Python loop if cohort is huge.
        # Ideally, use polars plugins, but sticking to Python logic from make_data.py
        filled = seq_df.group_by(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]).map_groups(fill_sequence_for_vent)
        labels = filled.group_by(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]).map_groups(per_stay_labels)
        
        answer = ds.join(labels, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], how="inner")
        
        self.registry.save_dataframe(answer, "dataset_123_answer.parquet")
        self.observer.log("INFO", f"LabelGenerator: Wrote dataset_123_answer.parquet (rows={answer.height})")
