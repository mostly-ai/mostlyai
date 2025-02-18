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
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.style import Style

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
            refresh_per_second=1 / interval,
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
    try:
        if progress_bar:
            # loop until job has completed
            progress.start()
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
                    current_task_id = progress_bars[step.id]
                    current_task = progress.tasks[current_task_id]
                    if not current_task.started and step.start_date is not None:
                        progress.start_task(current_task_id)
                    if step.progress.max > 0:
                        progress.update(
                            current_task_id,
                            total=step.progress.max,
                            completed=step.progress.value,
                        )
                    if current_task.started and step.end_date is not None:
                        progress.stop_task(current_task_id)
                    # break if step has failed or been canceled
                    if step.status in (ProgressStatus.failed, ProgressStatus.canceled):
                        rich.print(f"[red]Step {step.model_label} {step.step_code.value} {step.status.lower()}")
                        return
                # check whether we are done
                if job.progress.value >= job.progress.max:
                    progress.refresh()
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
            progress.stop()


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
