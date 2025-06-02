# Copyright 2024-2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Any, Union
from urllib.parse import urlparse

import pandas as pd
import rich
from rich import box
from rich.console import RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

from mostlyai.sdk.client._naming_conventions import map_camel_to_snake_case
from mostlyai.sdk.client.exceptions import APIError
from mostlyai.sdk.domain import (
    Generator,
    GeneratorListItem,
    ProgressStatus,
    StepCode,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
    SyntheticTableConfig,
    SyntheticTableConfiguration,
)


def check_local_mode_available() -> None:
    """
    Check if the local mode is available. Raise an exception if it is not.
    """
    try:
        from mostlyai.sdk._local.server import LocalServer  # noqa
        from mostlyai import qa  # noqa

        return
    except ImportError:
        raise APIError(
            "LOCAL mode requires additional packages to be installed. Run `pip install -U 'mostlyai[local]'`."
        )


def validate_base_url(base_url: str) -> None:
    """
    Check if the provided base URL is valid. Raise an exception if it is not.
    """
    base_url = str(base_url)
    if not base_url:
        raise APIError("Missing base URL.")
    parsed = urlparse(base_url)
    if not all([parsed.scheme, parsed.netloc]):
        raise APIError("Invalid base URL.")


class _DynamicRefreshThread(Thread):
    """A thread that calls live.refresh() at dynamic intervals, increasing the delay between refreshes over time.
    The refresh interval stays at initial_interval for the first fixed_time seconds,
    then gradually increases to max_interval over the next increasing_time seconds.
    """

    def __init__(
        self,
        live: "Live",
        initial_interval: float = 2.0,
        max_interval: float = 30.0,
        fixed_time: float = 30.0,
        increasing_time: float = 60.0,
    ) -> None:
        self.live = live
        self.done = Event()
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.fixed_time = fixed_time
        self.increasing_time = increasing_time
        self.current_interval = initial_interval
        self.start_time = None
        super().__init__(daemon=True)

    def stop(self) -> None:
        self.done.set()

    def _calculate_interval(self, elapsed_time: float) -> float:
        if elapsed_time <= self.fixed_time:
            return self.initial_interval
        growth_factor = min(1.0, (elapsed_time - self.fixed_time) / self.increasing_time)
        interval = self.initial_interval + growth_factor * (self.max_interval - self.initial_interval)
        return min(interval, self.max_interval)

    def run(self) -> None:
        self.start_time = time.time()
        while not self.done.is_set():
            # calculate next interval based on elapsed time
            elapsed_time = time.time() - self.start_time
            self.current_interval = self._calculate_interval(elapsed_time)
            # wait for the calculated interval or until stopped
            if self.done.wait(self.current_interval):
                break
            # refresh the display if not stopped
            with self.live._lock:
                if not self.done.is_set():
                    self.live.refresh()


class _LiveWithDynamicRefresh(Live):
    def __init__(self, *args, **kwargs):
        # disable auto refresh implemented in the parent class
        kwargs["auto_refresh"] = False
        super().__init__(*args, **kwargs)

    def start(self, refresh: bool = False) -> None:
        super().start(refresh)
        self._refresh_thread = _DynamicRefreshThread(self)
        self._refresh_thread.start()

    def stop(self) -> None:
        if self._refresh_thread is not None:
            self._refresh_thread.stop()
            self._refresh_thread = None
        return super().stop()


def job_wait(
    get_progress: Callable,
    interval: float,
    progress_bar: bool = True,
) -> None:
    # ensure that interval is at least 1 sec
    interval = max(interval, 1)
    # retrieve current JobProgress
    job = get_progress()
    if progress_bar:
        # initialize progress bars
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                style=Style(color="rgb(245,245,245)"),
                complete_style=Style(color="rgb(66,77,179)"),
                finished_style=Style(color="rgb(36,219,149)"),
                pulse_style=Style(color="rgb(245,245,245)"),
            ),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            auto_refresh=False,  # auto refresh will be handled by Live object
            expand=True,
        )
        progress_bars = {
            "overall": progress.add_task(
                description="[bold]Overall job progress[/b]",
                start=job.start_date is not None,
                completed=0,
                total=job.progress.max,
            )
        }
        for step in job.steps:
            step_code = step.step_code.value
            if step_code == StepCode.train_model.value:
                step_code += " :gem:"
            progress_bars |= {
                step.id: progress.add_task(
                    description=f"Step {step.model_label or 'common'} [#808080]{step_code}[/]",
                    start=step.start_date is not None,
                    completed=0,
                    total=step.progress.max,
                )
            }
        layout = Table.grid(expand=True)
        layout.add_row(progress)
        live = _LiveWithDynamicRefresh(layout)
        step_id_to_layout_idx = {}

    def _rich_table_delete_row(table: Table, idx: int = -1) -> Table:
        # helper method to delete a row from a rich table
        for column in table.columns:
            column._cells = column._cells[:idx] + column._cells[idx + 1 :]
        table.rows = table.rows[:idx] + table.rows[idx + 1 :]
        return table

    def _rich_table_insert_row(*renderables: RenderableType | None, table: Table, idx: int = -1) -> Table:
        # helper method to insert a row into a rich table
        table.add_row(*renderables)
        for column in table.columns:
            column._cells.insert(idx, column._cells[-1])
            column._cells.pop()
        table.rows.insert(idx, table.rows[-1])
        table.rows.pop()
        return table

    try:
        if progress_bar:
            # loop until job has completed
            live.start()
        while True:
            # sleep for interval seconds
            time.sleep(interval)
            # retrieve current JobProgress
            job = get_progress()
            if progress_bar:
                current_task_id = progress_bars["overall"]
                current_task = progress.tasks[current_task_id]
                if not current_task.started and job.start_date is not None:
                    progress.start_task(current_task_id)
                # update progress bars
                progress.update(
                    current_task_id,
                    total=job.progress.max,
                    completed=job.progress.value,
                )
                if current_task.started and job.end_date is not None:
                    progress.stop_task(current_task_id)

                for i, step in enumerate(job.steps):
                    # create the latest training log table
                    if step.step_code == StepCode.train_model:
                        messages = step.messages or []
                        messages = messages[-6:]
                        last_checkpoint_idx = next(
                            (len(messages) - 1 - j for j, msg in enumerate(reversed(messages)) if msg["is_checkpoint"]),
                            -1,
                        )
                        training_log = Table(
                            title=f"Training log for `{step.model_label}`",
                            box=box.SIMPLE_HEAD,
                            expand=True,
                            header_style="none",
                        )
                        columns = ["Epochs", "Samples", "Elapsed Time", "Val Loss"]
                        if messages and messages[0].get("dp_eps"):
                            columns.append("Diff Privacy (ε/δ)")
                        for col in columns:
                            training_log.add_column(col, justify="right")
                        for j, message in enumerate(messages):
                            formatted_message = [
                                f"{message['epoch']:.2f}" if message.get("epoch") else "-",
                                f"{message['samples']:,}" if message.get("samples") else "-",
                                f"{message['total_time']:.0f}s" if message.get("total_time") else "-",
                                f"{message['val_loss']:.4f}" if message.get("val_loss") else "-",
                            ]
                            if message.get("dp_eps"):
                                formatted_message += [f"{message['dp_eps']:.2f} / {message['dp_delta']:.0e}"]
                            style = "#14b57d on #f0fff7" if j == last_checkpoint_idx else "bright_black"
                            training_log.add_row(*formatted_message, style=style)
                    current_task_id = progress_bars[step.id]
                    current_task = progress.tasks[current_task_id]
                    if not current_task.started and step.start_date is not None:
                        progress.start_task(current_task_id)
                    if step.step_code == StepCode.train_model and step_id_to_layout_idx.get(step.id) is None:
                        layout.add_row(Text("\n\n"))
                        layout.add_row(training_log)
                        step_id_to_layout_idx[step.id] = len(layout.rows) - 1
                    if step.progress.max > 0:
                        progress.update(
                            current_task_id,
                            total=step.progress.max,
                            completed=step.progress.value,
                        )
                    if current_task.started:
                        if step.step_code == StepCode.train_model:
                            _rich_table_delete_row(layout, step_id_to_layout_idx[step.id])
                            _rich_table_insert_row(training_log, table=layout, idx=step_id_to_layout_idx[step.id])
                        if step.end_date is not None:
                            progress.stop_task(current_task_id)
                    live.update(layout)
                    # break if step has failed or been canceled
                    if step.status in (ProgressStatus.failed, ProgressStatus.canceled):
                        rich.print(f"[red]Step {step.model_label} {step.step_code.value} {step.status.lower()}")
                        return
                # check whether we are done
                if job.progress.value >= job.progress.max:
                    live.refresh()
                    time.sleep(1)  # give the system a moment to update the status
                    return
            else:
                if job.end_date or job.progress in (
                    ProgressStatus.failed,
                    ProgressStatus.canceled,
                ):
                    rich.print(f"Job {job.status.lower()}")
                    return
    except KeyboardInterrupt:
        rich.print(f"[red]Step {step.model_label} {step.step_code.value} {step.status.lower()}")
        return
    finally:
        if progress_bar:
            live.stop()


def get_subject_table_names(generator: Generator) -> list[str]:
    subject_tables = []
    for table in generator.tables:
        ctx_fks = [fk for fk in table.foreign_keys or [] if fk.is_context]
        if len(ctx_fks) == 0:
            subject_tables.append(table.name)
    return subject_tables


Seed = Union[pd.DataFrame, str, Path, list[dict[str, Any]]]


def harmonize_sd_config(
    generator: Generator | str | None = None,
    get_generator: Callable[[str], Generator] | None = None,
    size: int | dict[str, int] | None = None,
    seed: Seed | dict[str, Seed] | None = None,
    config: SyntheticDatasetConfig | SyntheticProbeConfig | dict | None = None,
    config_type: (type[SyntheticDatasetConfig] | type[SyntheticProbeConfig] | None) = None,
    name: str | None = None,
) -> SyntheticDatasetConfig | SyntheticProbeConfig:
    config_type = config_type or SyntheticDatasetConfig
    if config is None:
        config = config_type()
    elif isinstance(config, dict):
        config = map_camel_to_snake_case(config)
        config = config_type(**config)

    size = size if size is not None else {}
    seed = seed if seed is not None else {}

    if isinstance(generator, GeneratorListItem):
        generator = get_generator(generator.id)
    if isinstance(generator, Generator):
        generator_id = str(generator.id)
    elif generator is not None:
        generator_id = str(generator)
        generator = get_generator(generator_id)
    elif config.generator_id:
        generator_id = config.generator_id
        generator = get_generator(generator_id)
    else:
        raise ValueError("Either a generator or a configuration with a generator_id must be provided.")
    config.generator_id = generator_id

    if not isinstance(size, dict) or not isinstance(seed, dict) or not config.tables:
        subject_tables = get_subject_table_names(generator)
    else:
        subject_tables = []

    # normalize size
    if not isinstance(size, dict):
        size = {table: size for table in subject_tables}

    # normalize seed, applicable only for the first subject table
    if not isinstance(seed, dict):
        seed = {table: seed for table in subject_tables[:1]}

    # insert name into config
    if name is not None:
        config.name = name

    def size_and_seed_table_configuration(table_name):
        return SyntheticTableConfiguration(
            sample_size=size.get(table_name),
            sample_seed_data=seed.get(table_name) if not isinstance(seed.get(table_name), list) else None,
            sample_seed_dict=pd.DataFrame(seed.get(table_name)) if isinstance(seed.get(table_name), list) else None,
        )

    # infer tables if not provided
    if not config.tables:
        config.tables = []
        for table in generator.tables:
            configuration = size_and_seed_table_configuration(table.name)
            config.tables.append(SyntheticTableConfig(name=table.name, configuration=configuration))
    else:
        for table in config.tables:
            configuration = size_and_seed_table_configuration(table.name)
            table.configuration.sample_size = table.configuration.sample_size or configuration.sample_size
            table.configuration.sample_seed_data = (
                table.configuration.sample_seed_data or configuration.sample_seed_data
            )
            table.configuration.sample_seed_dict = (
                table.configuration.sample_seed_dict or configuration.sample_seed_dict
            )

    return config
