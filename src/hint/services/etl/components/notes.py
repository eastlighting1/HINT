import re
import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig


class NoteTokenizer(PipelineComponent):
    """Split clinical notes into sentences for downstream text processing.

    Parses raw NOTEEVENTS, respects ICU windows, and emits sentence-level rows for NLP tasks.

    Attributes:
        cfg (ETLConfig): ETL configuration containing source directories.
        registry (Registry): Registry placeholder for interface alignment.
        observer (TelemetryObserver): Telemetry adapter for logging.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer
        self.stopwords = set(["of", "on", "or"])

    def execute(self) -> None:
        """Tokenize notes into sentence-level records.

        Returns:
            None

        Raises:
            FileNotFoundError: If expected raw note files are missing.
        """
        try:
            import spacy
            _ = spacy
        except ImportError:
            self.observer.log("WARNING", "NoteTokenizer: spaCy not installed, skipping note tokenization.")
            return

        raw_dir = Path(self.cfg.raw_dir)
        proc_dir = Path(self.cfg.proc_dir)

        self.observer.log("INFO", "NoteTokenizer: Loading NOTEEVENTS and ICU boundaries")

        notes = (
            pl.read_csv(raw_dir / "NOTEEVENTS.csv.gz", infer_schema_length=0)
            .with_columns([pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("CHARTTIME")])
            .filter(pl.col("ISERROR").is_null())
        )

        icu = (
            pl.read_csv(raw_dir / "ICUSTAYS.csv.gz", infer_schema_length=0)
            .with_columns([pl.col("OUTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("OUTTIME")])
            .select(["HADM_ID", "ICUSTAY_ID", "OUTTIME"])
        )

        joined = notes.join(icu, on="HADM_ID", how="left")
        joined = joined.filter(pl.col("CHARTTIME") <= pl.col("OUTTIME"))

        self.observer.log("INFO", f"NoteTokenizer: Notes after ICU window filter rows={joined.height}")

        def is_title(text: str) -> bool:
            if not text.endswith(":"):
                return False
            t = text[:-1]
            t = re.sub(r"(\([^\)]*?\))", "", t)
            for word in t.split():
                if word in self.stopwords:
                    continue
                if not word[0].isupper():
                    return False
            if t == "Disp":
                return False
            return True

        def is_inline_title(text: str) -> bool:
            m = re.search(r"^([a-zA-Z ]+:) ", text)
            if not m:
                return False
            return is_title(m.groups()[0])

        def sent_tokenize_rules(text: str) -> List[str]:
            text = re.sub(r"---+", "\n\n-----\n\n", text)
            text = re.sub(r"___+", "\n\n_____\n\n", text)
            text = re.sub(r"\n\n+", "\n\n", text)
            segments = text.split("\n\n")
            new_segments: List[str] = []

            m1 = re.match(r"(Admission Date:) (.*) (Discharge Date:) (.*)", segments[0] if segments else "")
            if m1:
                new_segments += [s.strip() for s in m1.groups()]
                segments = segments[1:]

            m2 = re.match(r"(Date of Birth:) (.*) (Sex:) (.*)", segments[0] if segments else "")
            if m2:
                new_segments += [s.strip() for s in m2.groups()]
                segments = segments[1:]

            for segment in segments:
                possible_headers = re.findall(r"\n([A-Z][^\n:]+:)", "\n" + segment)
                headers = [h.strip() for h in possible_headers if is_title(h.strip())]
                for h in headers:
                    try:
                        ind = segment.index(h)
                        prefix, rest = segment[:ind].strip(), segment[ind + len(h) :].strip()
                        if prefix:
                            new_segments.append(prefix)
                        new_segments.append(h)
                        segment = rest
                    except ValueError:
                        continue
                if segment:
                    new_segments.append(segment)
            segments = new_segments

            new_segments = []
            for segment in segments:
                parts = segment.split("\n_____\n")
                new_segments.append(parts[0])
                for ss in parts[1:]:
                    new_segments += ["_____", ss]
            segments = new_segments

            new_segments = []
            for segment in segments:
                parts = segment.split("\n-----\n")
                new_segments.append(parts[0])
                for ss in parts[1:]:
                    new_segments += ["-----", ss]
            segments = new_segments

            new_segments = []
            if segments:
                for segment in segments:
                    if not re.search(r"\n\s*\d+\.", "\n" + segment):
                        new_segments.append(segment)
                        continue
                    seg = "\n" + segment
                    m = re.search(r"\n\s*(\d+)\.", seg)
                    if not m:
                        new_segments.append(segment)
                        continue
                    start = int(m.group(1))
                    n = start
                    while re.search(r"\n\s*%d\." % n, seg):
                        n += 1
                    n -= 1
                    if n <= start:
                        new_segments.append(segment)
                        continue
                    for i in range(start, n + 1):
                        match_obj = re.search(r"\n\s*%d\." % i, seg)
                        if not match_obj:
                            continue
                        match = match_obj.group(0)
                        try:
                            idx = seg.index(match)
                            prefix, seg = seg[:idx].strip(), seg[idx:].strip()
                            if prefix:
                                new_segments.append(prefix)
                        except ValueError:
                            continue
                    if seg:
                        new_segments.append(seg)
            segments = new_segments

            new_segments = []
            for segment in segments:
                lines = segment.split("\n")
                buf: List[str] = []
                for line in lines:
                    if is_inline_title(line):
                        if buf:
                            new_segments.append("\n".join(buf))
                        buf = []
                    buf.append(line)
                if buf:
                    new_segments.append("\n".join(buf))
            segments = new_segments

            new_segments = []
            n_segments = len(segments)
            for i, seg in enumerate(segments):
                if i > 0 and "\n" not in seg and is_title(segments[i - 1]) and (i == n_segments - 1 or is_title(segments[i + 1])):
                    if new_segments:
                        new_segments[-1] = segments[i - 1] + " " + seg
                    else:
                        new_segments.append(seg)
                else:
                    new_segments.append(seg)

            return new_segments

        rows: List[Dict[str, Any]] = []
        total_rows = joined.height
        for idx, r in enumerate(joined.iter_rows(named=True)):
            sents = sent_tokenize_rules(r["TEXT"] or "")
            for i, s in enumerate(sents):
                rows.append(
                    {
                        "SUBJECT_ID": r["SUBJECT_ID"],
                        "HADM_ID": r["HADM_ID"],
                        "ICUSTAY_ID": r.get("ICUSTAY_ID"),
                        "CHARTTIME": r.get("CHARTTIME"),
                        "CATEGORY": r.get("CATEGORY"),
                        "DESCRIPTION": r.get("DESCRIPTION"),
                        "SENTENCE_ID": i,
                        "SENTENCE": s,
                    }
                )

            if idx % 10000 == 0:
                self.observer.log("DEBUG", f"NoteTokenizer: Processed {idx}/{total_rows} notes")

        out_path = proc_dir / "notes_sentences.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(out_path)
        self.observer.log("INFO", f"NoteTokenizer: Wrote {len(rows)} sentences to {out_path}")
