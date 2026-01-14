"""Summary of the timeseries module.

Longer description of the module purpose and usage.
"""

import polars as pl

from pathlib import Path

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig



class TimeSeriesAggregator(PipelineComponent):

    """Summary of TimeSeriesAggregator purpose.
    
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

        raw_dir = Path(self.cfg.raw_dir)

        resources_dir = Path(self.cfg.resources_dir)

        proc_dir = Path(self.cfg.proc_dir)

        proc_dir.mkdir(parents=True, exist_ok=True)





        progress = getattr(self.observer, "get_child_progress", lambda: None)()

        task_id = None

        if progress:

            task_id = progress.add_task("[cyan]TimeSeries: Init", total=4)





        if progress: progress.update(task_id, description="[cyan]TimeSeries (1/4): Scan Raw Dir")

        self.observer.log("INFO", f"TimeSeriesAggregator: Stage 1/4 scanning raw directory {raw_dir}")



        self.observer.log("INFO", "TimeSeriesAggregator: Loading variable map")

        varmap = pl.read_csv(resources_dir / "itemid_to_variable_map.csv").with_columns(

            pl.col("ITEMID").cast(pl.Int64)

        ).select([

            "ITEMID",

            pl.col("MIMIC LABEL").alias("LABEL"),

            pl.col("LEVEL1"),

            pl.col("LEVEL2")

        ]).lazy()



        ranges_path = resources_dir / "variable_ranges.csv"

        var_ranges = None

        if ranges_path.exists():

            self.observer.log("INFO", f"TimeSeriesAggregator: Loading variable ranges from {ranges_path.name}")

            var_ranges = pl.read_csv(ranges_path).select([

                pl.col("LEVEL2").str.to_lowercase().alias("LEVEL2"),

                pl.col("OUTLIER LOW").alias("OUT_LOW"),

                pl.col("OUTLIER HIGH").alias("OUT_HIGH"),

                pl.col("VALID LOW").alias("VAL_LOW"),

                pl.col("VALID HIGH").alias("VAL_HIGH")

            ])

        else:

            self.observer.log("WARNING", "TimeSeriesAggregator: Variable ranges not found; skipping outlier filtering.")



        self.observer.log("INFO", "TimeSeriesAggregator: Loading ICU stay map")

        icu_map = pl.scan_csv(raw_dir / "ICUSTAYS.csv.gz", infer_schema_length=0, has_header=True).with_columns([

            pl.col("ICUSTAY_ID").cast(pl.Int64),

            pl.col("HADM_ID").cast(pl.Int64),

            self._to_datetime_iso("INTIME").alias("INTIME_DT"),

            self._to_datetime_iso("OUTTIME").alias("OUTTIME_DT"),

        ]).select(["ICUSTAY_ID", "HADM_ID", "INTIME_DT", "OUTTIME_DT"]).unique(subset=["ICUSTAY_ID"])



        if progress: progress.advance(task_id)





        if progress: progress.update(task_id, description="[cyan]TimeSeries (2/4): Load Events")



        def process_events(table: str, time_col: str) -> pl.LazyFrame:

            """Summary of process_events.
            
            Longer description of the process_events behavior and usage.
            
            Args:
            table (Any): Description of table.
            time_col (Any): Description of time_col.
            
            Returns:
            pl.LazyFrame: Description of the return value.
            
            Raises:
            Exception: Description of why this exception might be raised.
            """

            fpath = raw_dir / f"{table.upper()}.csv.gz"

            if not fpath.exists():

                fpath = raw_dir / f"{table.upper()}.csv"



            self.observer.log("INFO", f"TimeSeriesAggregator: Reading {table} from {fpath.name}")

            try:

                columns = pl.read_csv(fpath, n_rows=0).columns

            except Exception:

                self.observer.log("WARNING", f"Could not read header for {fpath}, assuming standard columns.")

                columns = []



            lf_raw = pl.scan_csv(fpath, infer_schema_length=0, has_header=True)



            casts = [

                pl.col("ITEMID").cast(pl.Int64),

                pl.col("SUBJECT_ID").cast(pl.Int64),

                pl.col("HADM_ID").cast(pl.Int64),

                pl.col("VALUENUM").cast(pl.Float64),

                self._to_datetime_iso(time_col.upper()).alias("EVENT_TS"),

            ]



            if "VALUEUOM" in columns:

                casts.append(pl.col("VALUEUOM"))

            else:

                casts.append(pl.lit(None).cast(pl.Utf8).alias("VALUEUOM"))



            if "ICUSTAY_ID" in columns:

                casts.append(pl.col("ICUSTAY_ID").cast(pl.Int64))



            ev = lf_raw.with_columns(casts)



            ev = ev.filter(pl.col("VALUENUM").is_not_null())



            ev_labeled = ev.join(varmap, on="ITEMID", how="inner")



            if "ICUSTAY_ID" in columns:

                self.observer.log("INFO", f"TimeSeriesAggregator: Joining {table} directly on ICUSTAY_ID")

                joined = ev_labeled.join(icu_map, on="ICUSTAY_ID", how="inner")

                joined = joined.filter(

                    (pl.col("EVENT_TS") >= pl.col("INTIME_DT")) &

                    (pl.col("EVENT_TS") <= pl.col("OUTTIME_DT"))

                )

            else:

                self.observer.log("INFO", f"TimeSeriesAggregator: Mapping {table} via HADM_ID (with 6h lookback)")



                lookback = pl.duration(hours=6)



                joined = ev_labeled.join(icu_map, on="HADM_ID", how="inner").filter(

                    (pl.col("EVENT_TS") >= (pl.col("INTIME_DT") - lookback)) &

                    (pl.col("EVENT_TS") <= pl.col("OUTTIME_DT"))

                )



            return joined.with_columns([

                ((pl.col("EVENT_TS").dt.epoch("us") - pl.col("INTIME_DT").dt.epoch("us")) / 3.6e9).floor().cast(pl.Int32).alias("HOURS_IN")

            ]).filter(
                pl.col("HOURS_IN") >= 0
            ).select(["ICUSTAY_ID", "HOURS_IN", "LEVEL1", "LEVEL2", "VALUENUM", "VALUEUOM"])



        self.observer.log("INFO", "TimeSeriesAggregator: Loading chart events")

        lf_chart = process_events("chartevents", "charttime")

        self.observer.log("INFO", "TimeSeriesAggregator: Loading lab events")

        lf_lab = process_events("labevents", "charttime")



        full_lf = pl.concat([lf_chart, lf_lab])



        if progress: progress.advance(task_id)





        if progress: progress.update(task_id, description="[cyan]TimeSeries (3/4): Convert Units")

        self.observer.log("INFO", "TimeSeriesAggregator: Stage 2/4 applying unit conversions")



        is_weight = pl.col("LEVEL1").str.to_lowercase().str.contains("weight")

        is_temp = pl.col("LEVEL1").str.to_lowercase().str.contains("temperature")

        is_height = pl.col("LEVEL1").str.to_lowercase().str.contains("height")

        is_fio2 = pl.col("LEVEL1").str.to_lowercase().str.contains("fraction inspired oxygen")

        is_o2sat = pl.col("LEVEL1").str.to_lowercase().str.contains("oxygen saturation")



        uom = pl.col("VALUEUOM").str.to_lowercase()

        val = pl.col("VALUENUM")



        converted_lf = full_lf.with_columns(

            pl.when(is_weight & (uom.str.contains("oz")))

            .then(val / 16.0 * 0.45359237)

            .when(is_weight & (uom.str.contains("lb")))

            .then(val * 0.45359237)

            .when(is_temp & (uom.str.contains("f") | (val > 79)))

            .then((val - 32) * 5.0 / 9.0)

            .when(is_height & (uom.str.contains("in")))

            .then(val * 2.54)

            .when(is_fio2 & (val <= 1.0))

            .then(val * 100.0)

            .when(is_o2sat & (val <= 1.0))

            .then(val * 100.0)

            .otherwise(val)

            .alias("VALUENUM")

        )



        self.observer.log("INFO", "TimeSeriesAggregator: Stage 3/4 applying variable limits (outliers)")



        if var_ranges is not None:

            converted_lf = converted_lf.with_columns(pl.col("LEVEL2").str.to_lowercase().alias("L2_LOWER"))



            ranges_df = var_ranges

            joined_ranges = converted_lf.join(ranges_df.lazy(), left_on="L2_LOWER", right_on="LEVEL2", how="left")



            final_vals = (

                joined_ranges

                .with_columns([

                    (pl.col("VALUENUM") < pl.col("OUT_LOW")).fill_null(False).alias("IS_LO_OUT"),

                    (pl.col("VALUENUM") > pl.col("OUT_HIGH")).fill_null(False).alias("IS_HI_OUT"),

                    pl.col("VAL_LOW"),

                    pl.col("VAL_HIGH")

                ])

                .filter(~(pl.col("IS_LO_OUT") | pl.col("IS_HI_OUT")))

                .with_columns(

                    pl.when(pl.col("VALUENUM") < pl.col("VAL_LOW"))

                    .then(pl.col("VAL_LOW"))

                    .when(pl.col("VALUENUM") > pl.col("VAL_HIGH"))

                    .then(pl.col("VAL_HIGH"))

                    .otherwise(pl.col("VALUENUM"))

                    .alias("VALUENUM")

                )

            )

        else:

            final_vals = converted_lf



        if progress: progress.advance(task_id)





        if progress: progress.update(task_id, description="[cyan]TimeSeries (4/4): Aggregate")

        self.observer.log("INFO", "TimeSeriesAggregator: Stage 4/4 aggregating hourly statistics")



        vitals_labs = final_vals.group_by(["ICUSTAY_ID", "HOURS_IN", "LEVEL2"]).agg([

            pl.col("VALUENUM").mean().alias("MEAN"),

            pl.col("VALUENUM").count().alias("COUNT"),

            pl.col("VALUENUM").std().alias("STDDEV"),

        ]).collect()



        vitals_labs = vitals_labs.rename({"LEVEL2": "LABEL"})



        self.observer.log("INFO", "TimeSeriesAggregator: Writing vitals outputs")

        vitals_labs.write_parquet(proc_dir / self.cfg.artifacts.vitals_file)

        vitals_labs.select(["ICUSTAY_ID", "HOURS_IN", "LABEL", "MEAN"]).write_parquet(proc_dir / self.cfg.artifacts.vitals_mean_file)

        self.observer.log("INFO", "TimeSeriesAggregator: Finished")



        if progress:

            progress.update(task_id, visible=False)



    @staticmethod

    def _to_datetime_iso(col: str) -> pl.Expr:

        """Summary of _to_datetime_iso.
        
        Longer description of the _to_datetime_iso behavior and usage.
        
        Args:
        col (Any): Description of col.
        
        Returns:
        pl.Expr: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        base = pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)

        return pl.when(base.is_null() & pl.col(col).is_not_null()).then(

            pl.col(col).str.replace(r"Z$", "+00:00", literal=False).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)

        ).otherwise(base)
