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

from mostlyai.sdk._local.execution.plan import (
    FINALIZE_TRAINING_TASK_STEPS,
    TRAINING_TASK_REPORT_STEPS,
    TRAINING_TASK_STEPS,
    has_language_model,
    has_non_context_relationships,
    has_tabular_model,
)
from mostlyai.sdk._local.storage import (
    get_model_label,
    read_generator_from_json,
    write_connector_to_json,
    write_generator_to_json,
    write_job_progress_to_json,
)
from mostlyai.sdk.client._base_utils import convert_to_df
from mostlyai.sdk.domain import (
    Connector,
    ConnectorAccessType,
    ConnectorType,
    Generator,
    GeneratorConfig,
    JobProgress,
    ModelType,
    ProgressStatus,
    ProgressStep,
    ProgressValue,
    SourceColumnConfig,
    SourceForeignKeyConfig,
    SourceTableConfig,
    TaskType,
)


def create_generator(home_dir: Path, config: GeneratorConfig) -> Generator:
    # handle file uploads -> create_connectors
    for t in config.tables or []:
        if t.data is not None:
            connector = Connector(
                **{
                    "name": "FILE_UPLOAD",
                    "type": ConnectorType.file_upload,
                    "access_type": ConnectorAccessType.source,
                }
            )
            df = convert_to_df(data=t.data, format="parquet")
            fn = home_dir / "connectors" / connector.id / "data.parquet"
            fn.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(fn)
            t.data = None
            t.source_connector_id = connector.id
            t.location = str(fn.absolute())
            if t.columns is None:
                t.columns = [
                    SourceColumnConfig(
                        name=c,
                    )
                    for c in list(df.columns)
                ]
            write_connector_to_json(home_dir / "connectors" / connector.id, connector)

    # create generator
    generator = Generator(
        **{
            **config.model_dump(),
            "training_status": ProgressStatus.new,
            "tables": [
                {
                    **{k: v for k, v in t.model_dump().items() if k not in ["source_connector_id", "data"]},
                    "source_connector_id": t.source_connector_id,
                }
                for t in (config.tables or [])
            ],
        }
    )
    generator_dir = home_dir / "generators" / generator.id
    write_generator_to_json(generator_dir, generator)

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
            model_configuration = (
                table.tabular_model_configuration
                if model_type == ModelType.tabular
                else table.language_model_configuration
            )
            steps = TRAINING_TASK_STEPS + (
                TRAINING_TASK_REPORT_STEPS if model_configuration.enable_model_report else []
            )
            for step in steps:
                progress_steps.append(
                    ProgressStep(
                        task_type=TaskType.train_tabular
                        if model_type == ModelType.tabular
                        else TaskType.train_language,
                        model_label=get_model_label(table, model_type),
                        step_code=step,
                        progress=ProgressValue(value=0, max=1),
                        status=ProgressStatus.new,
                    )
                )
    if has_non_context_relationships(generator):
        for step in FINALIZE_TRAINING_TASK_STEPS:
            progress_steps.append(
                ProgressStep(
                    task_type=TaskType.finalize_training,
                    model_label=None,
                    step_code=step,
                    progress=ProgressValue(value=0, max=1),
                    status=ProgressStatus.new,
                )
            )
    job_progress = JobProgress(
        id=generator.id,
        progress=ProgressValue(value=0, max=len(progress_steps)),
        steps=progress_steps,
    )
    write_job_progress_to_json(generator_dir, job_progress)

    return generator


def get_generator_config(home_dir: Path, generator_id: str) -> GeneratorConfig:
    generator_dir = home_dir / "generators" / generator_id
    generator = read_generator_from_json(generator_dir)
    # construct GeneratorConfig explicitly to avoid validation warnings of extra fields
    config = GeneratorConfig(
        name=generator.name,
        description=generator.description,
        tables=[
            SourceTableConfig(
                name=t.name,
                source_connector_id=t.source_connector_id,
                location=t.location,
                tabular_model_configuration=t.tabular_model_configuration,
                language_model_configuration=t.language_model_configuration,
                primary_key=t.primary_key,
                foreign_keys=[SourceForeignKeyConfig.model_construct(**k.model_dump()) for k in t.foreign_keys]
                if t.foreign_keys
                else None,
                columns=[SourceColumnConfig.model_construct(**c.model_dump()) for c in t.columns if c.included]
                if t.columns
                else None,
            )
            for t in generator.tables
        ]
        if generator.tables
        else None,
    )
    return config
