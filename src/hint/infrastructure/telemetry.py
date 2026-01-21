"""Summary of the telemetry module.

Longer description of the module purpose and usage.
"""

from pathlib import Path

from typing import Optional, Dict, Iterable
import csv
import json
import time

from loguru import logger as loguru_logger

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout



from ..foundation.interfaces import TelemetryObserver





class RichTelemetryObserver(TelemetryObserver):

    """Summary of RichTelemetryObserver purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    _active_progress (Any): Description of _active_progress.
    console (Any): Description of console.
    console_logger (Any): Description of console_logger.
    file_logger (Any): Description of file_logger.
    metrics (Any): Description of metrics.
    """



    def __init__(self, run_dir: Path) -> None:

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.console = Console()

        self.metrics: Dict[str, list] = {}

        self.run_dir = Path(run_dir)

        self.logs_dir = self.run_dir / "logs"

        self.metrics_dir = self.run_dir / "metrics"

        self.traces_dir = self.run_dir / "traces"

        self.artifacts_dir = self.run_dir / "artifacts"

        for directory in [self.logs_dir, self.metrics_dir, self.traces_dir, self.artifacts_dir]:

            directory.mkdir(parents=True, exist_ok=True)

        self.file_logger = self._build_file_logger()

        self._active_progress: Optional[Progress] = None

        self._init_trace_files()

        self._dashboard_live: Optional[Live] = None
        self._dashboard_layout: Optional[Layout] = None
        self._dashboard_progress: Optional[Progress] = None
        self._dashboard_task_id: Optional[int] = None
        self._dashboard_started_at: Optional[float] = None
        self._dashboard_last_update: Optional[float] = None

        self._progress_live: Optional[Live] = None
        self._progress_layout: Optional[Layout] = None
        self._progress_bar: Optional[Progress] = None
        self._progress_task_id: Optional[int] = None


    def _event_markers(self) -> Iterable[str]:

        return (
            "[STAGE",
            "Stage",
            "Pipeline",
            "EPOCH",
            "checkpoint",
            "CHECKPOINT",
            "RUN END",
            "App:",
            "ETL Service",
            "ICD",
            "Intervention",
        )

    def _init_trace_files(self) -> None:

        execution_path = self.traces_dir / "execution.jsonl"

        bottleneck_path = self.traces_dir / "bottlenecks.log"

        if not execution_path.exists():

            execution_path.write_text("", encoding="utf-8")

        if not bottleneck_path.exists():

            bottleneck_path.write_text("", encoding="utf-8")


    def trace_event(self, name: str, payload: Dict[str, object]) -> None:

        """Append a JSONL trace event."""

        record = {"ts": time.time(), "name": name}

        record.update(payload)

        path = self.traces_dir / "execution.jsonl"

        with path.open("a", encoding="utf-8") as handle:

            handle.write(json.dumps(record) + "\n")


    def trace_bottleneck(self, name: str, duration_sec: float, threshold_sec: float) -> None:

        """Record a bottleneck entry when duration exceeds the threshold."""

        if duration_sec < threshold_sec:

            return

        path = self.traces_dir / "bottlenecks.log"

        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} duration_sec={duration_sec:.2f}\n"

        with path.open("a", encoding="utf-8") as handle:

            handle.write(line)



    def _build_file_logger(self):

        """Summary of _build_file_logger.
        
        Longer description of the _build_file_logger behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        loguru_logger.remove()

        system_log = self.logs_dir / "system.log"

        events_log = self.logs_dir / "events.log"

        loguru_logger.add(

            system_log,

            level="DEBUG",

            enqueue=True,

            format="[{time:YYYY-MM-DD HH:mm:ss}][{level}][{name}] {message}",

        )

        def event_filter(record) -> bool:

            level = record["level"].name

            if level in {"WARNING", "ERROR", "CRITICAL"}:

                return True

            message = record["message"]

            return any(marker in message for marker in self._event_markers())

        loguru_logger.add(

            events_log,

            level="INFO",

            enqueue=True,

            filter=event_filter,

            format="[{time:YYYY-MM-DD HH:mm:ss}][{level}][{name}] {message}",

        )

        return loguru_logger.bind(channel="hint")



    def log(self, level: str, message: str) -> None:

        """Summary of log.
        
        Longer description of the log behavior and usage.
        
        Args:
        level (Any): Description of level.
        message (Any): Description of message.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        normalized = level.upper()

        self.file_logger.log(normalized, message)



    def track_metric(self, name: str, value: float, step: int) -> None:

        """Summary of track_metric.
        
        Longer description of the track_metric behavior and usage.
        
        Args:
        name (Any): Description of name.
        value (Any): Description of value.
        step (Any): Description of step.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if name not in self.metrics:

            self.metrics[name] = []

        self.metrics[name].append({"step": step, "value": value})

        history_path = self.metrics_dir / "history.csv"

        write_header = not history_path.exists()

        with history_path.open("a", newline="") as handle:

            writer = csv.writer(handle)

            if write_header:

                writer.writerow(["name", "step", "value"])

            writer.writerow([name, step, value])



    def create_progress(self, desc: str, total: int) -> Progress:

        """Summary of create_progress.
        
        Longer description of the create_progress behavior and usage.
        
        Args:
        desc (Any): Description of desc.
        total (Any): Description of total.
        
        Returns:
        Progress: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        progress = Progress(

            SpinnerColumn(),

            TextColumn("[bold blue]{task.description}"),

            BarColumn(bar_width=40),

            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),

            TimeRemainingColumn(),

            console=self.console,

            transient=True,

        )

        self._active_progress = progress

        return progress



    def get_child_progress(self) -> Optional[Progress]:

        """Summary of get_child_progress.
        
        Longer description of the get_child_progress behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Optional[Progress]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return self._active_progress

    def render_stage0_receipt(self, context: object) -> None:

        """Render a static receipt-style summary for stage 0."""

        tree = Tree("Pipeline Receipt", guide_style="dim")

        if hasattr(context, "mode"):

            tree.add(f"mode: {getattr(context, 'mode')}")

        if hasattr(context, "seed"):

            tree.add(f"seed: {getattr(context, 'seed')}")

        tree.add(f"run_dir: {self.run_dir}")

        table = Table(show_header=False, box=None)

        table.add_row("logs", str(self.logs_dir))
        table.add_row("metrics", str(self.metrics_dir))
        table.add_row("traces", str(self.traces_dir))
        table.add_row("artifacts", str(self.artifacts_dir))

        panel = Panel.fit(tree, title="Stage 0: Initialization", border_style="blue")

        self.console.print(panel)

        self.console.print(table)

    def render_run_end(self, status: str, message: str) -> None:

        """Render a run-end panel with status."""

        style = "green" if status == "success" else "red"

        panel = Panel.fit(message, title=f"Run {status.upper()}", border_style=style)

        self.console.print(panel)


    def start_dashboard(self, title: str) -> None:

        """Start a live dashboard for training metrics."""

        if self._dashboard_live is not None:

            return

        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

        task_id = progress.add_task("Initializing...", total=1)

        layout["header"].update(Panel(title, border_style="cyan"))
        layout["body"].update(Panel("Waiting for metrics...", border_style="dim"))
        layout["footer"].update(progress)

        self._dashboard_layout = layout
        self._dashboard_progress = progress
        self._dashboard_task_id = task_id
        self._dashboard_started_at = time.time()
        self._dashboard_live = Live(layout, console=self.console, refresh_per_second=4)
        self._dashboard_live.start()


    def update_dashboard(
        self,
        stage: str,
        epoch: Optional[int],
        total_epochs: Optional[int],
        metrics: Dict[str, float],
        note: str = "",
    ) -> None:

        """Update the live dashboard content."""

        if self._dashboard_live is None or self._dashboard_layout is None:

            return

        now = time.time()
        if self._dashboard_last_update is not None and (now - self._dashboard_last_update) < 0.5:
            return
        self._dashboard_last_update = now

        elapsed = 0.0 if self._dashboard_started_at is None else now - self._dashboard_started_at

        epoch_text = ""
        if epoch is not None and total_epochs is not None:
            epoch_text = f"epoch {epoch}/{total_epochs}"
        elif epoch is not None:
            epoch_text = f"epoch {epoch}"

        header = f"{stage} {epoch_text}  elapsed={elapsed:.1f}s"
        if note:
            header = f"{header}  {note}"

        table = Table(box=None, show_header=False)
        for key, value in metrics.items():
            table.add_row(str(key), f"{value:.4f}" if isinstance(value, (int, float)) else str(value))

        self._dashboard_layout["header"].update(Panel(header, border_style="cyan"))
        self._dashboard_layout["body"].update(Panel(table, border_style="blue"))


    def reset_dashboard_progress(self, total: int, description: str) -> None:

        if self._dashboard_progress is None or self._dashboard_task_id is None:

            return

        self._dashboard_progress.reset(self._dashboard_task_id, total=total, completed=0)
        self._dashboard_progress.update(self._dashboard_task_id, description=description)


    def advance_dashboard_progress(self, advance: int = 1) -> None:

        if self._dashboard_progress is None or self._dashboard_task_id is None:

            return

        self._dashboard_progress.advance(self._dashboard_task_id, advance)


    def stop_dashboard(self) -> None:

        if self._dashboard_live is None:

            return

        self._dashboard_live.stop()

        self._dashboard_live = None
        self._dashboard_layout = None
        self._dashboard_progress = None
        self._dashboard_task_id = None
        self._dashboard_started_at = None
        self._dashboard_last_update = None


    def start_progress_view(self, title: str, total: int) -> None:

        """Start a live progress view for ETL."""

        if self._progress_live is not None:
            return

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

        task_id = progress.add_task("Initializing...", total=total)
        layout["header"].update(Panel(title, border_style="green"))
        layout["body"].update(progress)
        layout["footer"].update(Panel("Starting...", border_style="dim"))

        self._progress_layout = layout
        self._progress_bar = progress
        self._progress_task_id = task_id
        self._progress_live = Live(layout, console=self.console, refresh_per_second=4)
        self._progress_live.start()


    def progress_update(self, description: str) -> None:

        if self._progress_bar is None or self._progress_task_id is None:
            return

        self._progress_bar.update(self._progress_task_id, description=description)


    def progress_advance(self, advance: int = 1) -> None:

        if self._progress_bar is None or self._progress_task_id is None:
            return

        self._progress_bar.advance(self._progress_task_id, advance)


    def progress_footer(self, message: str) -> None:

        if self._progress_layout is None:
            return

        self._progress_layout["footer"].update(Panel(message, border_style="dim"))


    def stop_progress_view(self) -> None:

        if self._progress_live is None:
            return

        self._progress_live.stop()

        self._progress_live = None
        self._progress_layout = None
        self._progress_bar = None
        self._progress_task_id = None
