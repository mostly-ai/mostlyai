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
import shutil
from pathlib import Path

from mostlyai.sdk._local.execution.plan import (
    FINALIZE_GENERATION_TASK_STEPS,
    get_model_type_generation_steps_map,
    has_language_model,
    has_tabular_model,
)
from mostlyai.sdk._local.storage import (
    get_model_label,
    read_generator_from_json,
    read_synthetic_dataset_from_json,
    write_connector_to_json,
    write_job_progress_to_json,
    write_synthetic_dataset_to_json,
)
from mostlyai.sdk.client._base_utils import convert_to_df
from mostlyai.sdk.domain import (
    Connector,
    ConnectorAccessType,
    ConnectorType,
    JobProgress,
    ModelType,
    ProgressStatus,
    ProgressStep,
    ProgressValue,
    SyntheticDataset,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
    SyntheticTable,
    SyntheticTableConfig,
    TaskType,
)


def create_synthetic_dataset(
    home_dir: Path,
    config: SyntheticDatasetConfig | SyntheticProbeConfig,
    sample_size: int | None = None,
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

    # validate & fill defaults in config against generator
    config.validate_against_generator(generator)

    # create synthetic tables
    sd_tables = []
    for g_table in generator.tables:
        sd_table = SyntheticTable(**next(t for t in config.tables if t.name == g_table.name).model_dump())
        sd_table.foreign_keys = g_table.foreign_keys
        sd_table.source_table_total_rows = g_table.total_rows
        sd_table.tabular_model_metrics = g_table.tabular_model_metrics
        sd_table.language_model_metrics = g_table.language_model_metrics
        # overwrite sample size of subject table, if provided
        is_subject = not any(fk.is_context for fk in g_table.foreign_keys or [])
        if is_subject and sample_size is not None:
            sd_table.configuration.sample_size = sample_size
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

    # copy ModelQA reports into synthetic dataset directory
    source_reports_dir = generator_dir / "ModelQAReports"
    dest_reports_dir = synthetic_dataset_dir / "ModelQAReports"
    if source_reports_dir.exists():
        shutil.copytree(source_reports_dir, dest_reports_dir)

    # create job progress
    progress_steps: list[ProgressStep] = []
    for table in generator.tables:
        sd_table = next(t for t in config.tables if t.name == table.name)
        steps_map = get_model_type_generation_steps_map(
            enable_data_report=sd_table.configuration.enable_data_report,
            table=table,
        )
        model_types = [
            model_type
            for model_type, check in [
                (ModelType.tabular, has_tabular_model(table)),
                (ModelType.language, has_language_model(table)),
            ]
            if check
        ]
        for model_type in model_types:
            for step in steps_map[model_type]:
                progress_steps.append(
                    ProgressStep(
                        task_type=TaskType.generate,
                        model_label=get_model_label(table, model_type),
                        step_code=step,
                        progress=ProgressValue(value=0, max=1),
                        status=ProgressStatus.new,
                    )
                )
    for step in FINALIZE_GENERATION_TASK_STEPS:
        progress_steps.append(
            ProgressStep(
                task_type=TaskType.generate,
                model_label=None,
                step_code=step,
                progress=ProgressValue(value=0, max=1),
                status=ProgressStatus.new,
            )
        )
    job_progress = JobProgress(
        id=synthetic_dataset.id,
        progress=ProgressValue(value=0, max=len(progress_steps)),
        steps=progress_steps,
    )
    write_job_progress_to_json(synthetic_dataset_dir, job_progress)
    return synthetic_dataset


def get_synthetic_dataset_config(home_dir: Path, synthetic_dataset_id: str) -> SyntheticDatasetConfig:
    synthetic_dataset_dir = home_dir / "synthetic-datasets" / synthetic_dataset_id
    synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
    # construct SyntheticDatasetConfig explicitly to avoid validation warnings of extra fields
    config = SyntheticDatasetConfig(
        generator_id=synthetic_dataset.generator_id,
        name=synthetic_dataset.name,
        description=synthetic_dataset.description,
        tables=[SyntheticTableConfig.model_construct(**t.model_dump()) for t in synthetic_dataset.tables]
        if synthetic_dataset.tables
        else None,
        delivery=synthetic_dataset.delivery,
    )
    return config
