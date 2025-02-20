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
from pathlib import Path
from typing import Union, Any
from collections.abc import Callable

import pandas as pd
import rich
from rich import box
from rich.console import RenderableType
from rich.live import Live
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

from mostlyai.sdk.domain import (
    StepCode,
    ProgressStatus,
    Generator,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
    SyntheticTableConfiguration,
    SyntheticTableConfig,
    Connector,
    SyntheticDataset,
    GeneratorListItem,
)
from mostlyai.sdk.client._naming_conventions import map_camel_to_snake_case


## utils for manipulating rich's Table object
def _delete_row(table: Table, idx: int = -1) -> Table:
    for column in table.columns:
        column._cells = column._cells[:idx] + column._cells[idx + 1 :]
    table.rows = table.rows[:idx] + table.rows[idx + 1 :]
    return table


def _insert_row(*renderables: RenderableType | None, table: Table, idx: int = -1) -> Table:
    table.add_row(*renderables)
    for column in table.columns:
        column._cells.insert(idx, column._cells[-1])
        column._cells.pop()
    table.rows.insert(idx, table.rows[-1])
    table.rows.pop()
    return table


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
        live = Live(layout, refresh_per_second=1 / interval)
        step_id_to_layout_idx = {}
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
                            title=f"Latest training log for `{step.model_label}`",
                            header_style="grey19",
                            box=box.SQUARE_DOUBLE_HEAD,
                            expand=True,
                        )
                        columns = ["Epochs", "Samples", "Elapsed time", "Val loss"]
                        if messages and messages[0].get("dp_eps"):
                            columns.append("Diff Privacy (ε/δ)")
                        for col in columns:
                            training_log.add_column(col, justify="right")
                        for j, message in enumerate(messages):
                            formatted_message = [
                                str(message[k]) for k in ["epoch", "samples", "total_time", "val_loss"]
                            ]
                            if message.get("dp_eps"):
                                formatted_message.append(f"{message['dp_eps']:.2f} / {message['dp_delta']:.1e}")
                            # TODO: check the actual HEX colors used in the platform
                            style = "#54b281 on #f3fff7" if j == last_checkpoint_idx else "bright_black"
                            training_log.add_row(*formatted_message, style=style)
                    current_task_id = progress_bars[step.id]
                    current_task = progress.tasks[current_task_id]
                    if not current_task.started and step.start_date is not None:
                        progress.start_task(current_task_id)
                        if step.step_code == StepCode.train_model:
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
                            _delete_row(layout, step_id_to_layout_idx[step.id])
                            _insert_row(training_log, table=layout, idx=step_id_to_layout_idx[step.id])
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


def _get_subject_table_names(generator: Generator) -> list[str]:
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
        subject_tables = _get_subject_table_names(generator)
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


ShareableResource = Union[Connector, Generator, SyntheticDataset]
