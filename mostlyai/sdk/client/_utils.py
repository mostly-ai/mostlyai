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
from tqdm import tqdm
from colorama import Fore, Style

from mostlyai.sdk.client._base_utils import convert_to_base64, read_table_from_path
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
    interval = max(interval, 1)  # Ensure interval is at least 1 second
    job = get_progress()

    if progress_bar:
        overall_pbar = tqdm(
            total=job.progress.max,
            desc=f"{Fore.WHITE}Overall job progress{Style.RESET_ALL}",
            unit="step",
            dynamic_ncols=True,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL),
        )
        step_pbars = {}

    try:
        while True:
            time.sleep(interval)  # Sleep for interval seconds
            job = get_progress()

            if progress_bar:
                overall_pbar.total = job.progress.max
                overall_pbar.n = job.progress.value
                overall_pbar.refresh()

                for step in job.steps:
                    step_code = step.step_code.value
                    if step_code == StepCode.train_model.value:
                        step_code += " :gem:"

                    if step.id not in step_pbars:
                        step_pbars[step.id] = tqdm(
                            total=step.progress.max,
                            desc=f"{Fore.LIGHTBLACK_EX}Step {step.model_label or 'common'} {step_code}{Style.RESET_ALL}",
                            unit="step",
                            dynamic_ncols=True,
                            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Style.RESET_ALL),
                        )

                    step_pbar = step_pbars[step.id]
                    step_pbar.total = step.progress.max
                    step_pbar.n = step.progress.value
                    step_pbar.refresh()

                    if step.status in (ProgressStatus.failed, ProgressStatus.canceled):
                        print(
                            f"{Fore.RED}Step {step.model_label} {step.step_code.value} {step.status.lower()}{Style.RESET_ALL}"
                        )
                        return

                if job.progress.value >= job.progress.max:
                    time.sleep(1)  # Give the system a moment to update the status
                    return
            else:
                if job.end_date or job.progress in (ProgressStatus.failed, ProgressStatus.canceled):
                    print(f"Job {job.status.lower()}")
                    return
    except KeyboardInterrupt:
        print(f"{Fore.RED}Step {step.model_label} {step.step_code.value} {step.status.lower()}{Style.RESET_ALL}")
        return
    finally:
        if progress_bar:
            overall_pbar.close()
            for step_pbar in step_pbars.values():
                step_pbar.close()


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

    # normalize seed
    if not isinstance(seed, dict):
        seed = {table: seed for table in subject_tables}

    # insert name into config
    if name is not None:
        config.name = name

    # infer tables if not provided
    if not config.tables:
        config.tables = []
        for table in generator.tables:
            configuration = SyntheticTableConfiguration(
                sample_size=None,
                sample_seed_data=None,
                sample_seed_dict=None,
            )
            if table.name in subject_tables:
                configuration.sample_size = size.get(table.name)
                configuration.sample_seed_data = (
                    seed.get(table.name) if not isinstance(seed.get(table.name), list) else None
                )
                configuration.sample_seed_dict = (
                    seed.get(table.name) if isinstance(seed.get(table.name), list) else None
                )
            config.tables.append(SyntheticTableConfig(name=table.name, configuration=configuration))

    # convert `sample_seed_data` to base64-encoded Parquet files
    # convert `sample_seed_dict` to base64-encoded dictionaries
    for table in config.tables:
        if not table.configuration:
            continue
        if table.configuration.sample_seed_data is not None:
            if isinstance(table.configuration.sample_seed_data, pd.DataFrame):
                table.configuration.sample_seed_data = convert_to_base64(table.configuration.sample_seed_data)
            elif isinstance(table.configuration.sample_seed_data, (Path, str)):
                _, df = read_table_from_path(table.configuration.sample_seed_data)
                table.configuration.sample_seed_data = convert_to_base64(df)
                del df
            else:
                raise ValueError("sample_seed_data must be a DataFrame or a file path")
        if table.configuration.sample_seed_dict is not None:
            table.configuration.sample_seed_dict = convert_to_base64(
                table.configuration.sample_seed_dict, format="jsonl"
            )

    return config


ShareableResource = Union[Connector, Generator, SyntheticDataset]
