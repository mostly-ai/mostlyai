# Copyright 2025 MOSTLY AI
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

from pathlib import Path

from mostlyai.sdk._local.storage import (
    write_synthetic_dataset_to_json,
    write_job_progress_to_json,
    read_generator_from_json,
    write_connector_to_json,
)
from mostlyai.sdk._local.execution.plan import (
    has_tabular_model,
    has_language_model,
    GENERATION_TASK_STEPS,
    GENERATION_FINAL_STEPS,
)
from mostlyai.sdk.client._base_utils import convert_to_df
from mostlyai.sdk.domain import (
    SyntheticDatasetConfig,
    SyntheticDataset,
    ProgressStatus,
    ProgressStep,
    ModelType,
    ProgressValue,
    JobProgress,
    SyntheticTable,
    SyntheticTableConfiguration,
    TaskType,
    Connector,
    ConnectorType,
    ConnectorAccessType,
    SyntheticProbeConfig,
)


def create_synthetic_dataset(
    home_dir: Path,
    config: SyntheticDatasetConfig | SyntheticProbeConfig,
    size: int | dict[str, int] | None = None,
) -> SyntheticDataset:
    # create a FILE_UPLOAD connector and replace sample_seed_dict/sample_seed_data with sample_seed_connector_id
    for t in config.tables or []:
        seed = None
        if t.configuration.sample_seed_dict is not None:
            seed = convert_to_df(data=t.configuration.sample_seed_dict, format="jsonl")
        elif t.configuration.sample_seed_data is not None:
            seed = convert_to_df(data=t.configuration.sample_seed_data, format="parquet")
        if seed is not None:
            connector = Connector(
                **{
                    "name": "FILE_UPLOAD",
                    "type": ConnectorType.file_upload,
                    "access_type": ConnectorAccessType.source,
                }
            )
            fn = home_dir / "connectors" / connector.id / "seed.parquet"
            fn.parent.mkdir(parents=True, exist_ok=True)
            seed.to_parquet(fn)
            t.configuration.sample_seed_dict = None
            t.configuration.sample_seed_data = None
            t.configuration.sample_seed_connector_id = connector.id
            write_connector_to_json(home_dir / "connectors" / connector.id, connector)

    # get generator
    generator_dir = home_dir / "generators" / config.generator_id
    generator = read_generator_from_json(generator_dir)

    # fill sample sizes
    # if there's a seed (sample_seed_connector_id is set), sample size will be ignored by the engine
    size = size if size is not None else {}
    sd_tables = []
    for g_table in generator.tables:
        if g_table.name in (t.name for t in (config.tables or [])):
            sd_table = SyntheticTable(**next(t for t in config.tables if t.name == g_table.name).model_dump())
        else:
            sd_table = SyntheticTable(name=g_table.name)
        sd_table.foreign_keys = g_table.foreign_keys
        sd_table.source_table_total_rows = g_table.total_rows
        if sd_table.configuration is None:
            sd_table.configuration = SyntheticTableConfiguration()
        is_subject = not any(fk.is_context for fk in g_table.foreign_keys or [])
        if is_subject and sd_table.configuration.sample_size is None:
            if isinstance(size, int):
                sd_table.configuration.sample_size = size
            else:  # isinstance(size, dict)
                default_sample_size = 1 if isinstance(config, SyntheticProbeConfig) else g_table.total_rows
                sd_table.configuration.sample_size = size.get(g_table.name, default_sample_size)
        elif not is_subject:
            sd_table.configuration.sample_size = None  # sample size is not applicable to linked tables
        sd_tables.append(sd_table)

    # create synthetic dataset
    synthetic_dataset = SyntheticDataset(
        **{
            **config.model_dump(),
            "generation_status": ProgressStatus.new,
            "tables": sd_tables,
        }
    )
    synthetic_dataset.name = synthetic_dataset.name or generator.name
    synthetic_dataset.description = synthetic_dataset.description or generator.description
    synthetic_dataset_dir = home_dir / "synthetic-datasets" / synthetic_dataset.id
    write_synthetic_dataset_to_json(synthetic_dataset_dir, synthetic_dataset)

    # create job progress
    progress_steps: list[ProgressStep] = []
    for table in generator.tables:
        model_types = [
            model_type
            for model_type, check in [
                (ModelType.tabular, has_tabular_model(table)),
                (ModelType.language, has_language_model(table)),
            ]
            if check
        ]
        for model_type in model_types:
            for step in GENERATION_TASK_STEPS:
                progress_steps.append(
                    ProgressStep(
                        task_type=TaskType.generate_tabular
                        if model_type == ModelType.tabular
                        else TaskType.generate_language,
                        model_label=f"{table.name}:{model_type.value.lower()}",
                        step_code=step,
                        progress=ProgressValue(value=0, max=1),
                        status=ProgressStatus.new,
                    )
                )
    for step in GENERATION_FINAL_STEPS:
        progress_steps.append(
            ProgressStep(
                task_type=TaskType.finalize_generation,
                model_label=None,
                step_code=step,
                progress=ProgressValue(value=0, max=1),
                status=ProgressStatus.new,
            )
        )
    job_progress = JobProgress(
        progress=ProgressValue(value=0, max=len(progress_steps)),
        steps=progress_steps,
    )
    write_job_progress_to_json(synthetic_dataset_dir, job_progress)
    return synthetic_dataset