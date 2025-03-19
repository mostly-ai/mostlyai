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

import rich

from mostlyai.sdk._local import connectors
from mostlyai.sdk._local.storage import (
    write_generator_to_json,
    write_connector_to_json,
    write_job_progress_to_json,
    read_generator_from_json,
)
from mostlyai.sdk._local.execution.plan import (
    has_tabular_model,
    has_language_model,
    TRAINING_TASK_STEPS,
    TRAINING_TASK_REPORT_STEPS,
)
from mostlyai.sdk.client._base_utils import convert_to_df
from mostlyai.sdk.domain import (
    GeneratorConfig,
    ModelEncodingType,
    ProgressStatus,
    Generator,
    SourceColumnConfig,
    ConnectorAccessType,
    Connector,
    ConnectorType,
    ProgressStep,
    ModelType,
    ProgressValue,
    JobProgress,
    TaskType,
    SourceTableConfig,
    SourceForeignKeyConfig,
)


def create_generator(home_dir: Path, config: GeneratorConfig) -> Generator:
    # handle file uploads -> create_connectors
    for t in config.tables:
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
            write_connector_to_json(home_dir / "connectors" / connector.id, connector)

        # auto detection is only needed if columns are empty or have auto encoding types
        should_detect_schema = (t.columns is None) or any(
            col.model_encoding_type == ModelEncodingType.auto for col in (t.columns or [])
        )
        if should_detect_schema:
            table_schema = connectors.location_schema(connector, t.location)
            auto_detected_columns = {c.name: c.default_model_encoding_type for c in table_schema.columns}
            auto_detected_primary_key = table_schema.primary_key

            if t.columns is None:
                t.columns = [
                    SourceColumnConfig(
                        name=name,
                        model_encoding_type=enc_type,
                    )
                    for name, enc_type in auto_detected_columns.items()
                ]
                # summarize auto-detected encoding types
                encoding_types_counts = {}
                for enc_type in ModelEncodingType:
                    encoding_types_counts[enc_type.value] = 0
                for col in t.columns:
                    encoding_types_counts[col.model_encoding_type.value] += 1
                encoding_summary = ", ".join(
                    f"{count}x {enc_type}" for enc_type, count in encoding_types_counts.items() if count > 0
                )
                rich.print(f"Detected for Table `{t.name}` {encoding_summary} columns")
                foreign_keys = [fk.column for fk in t.foreign_keys or []]
                if (
                    auto_detected_primary_key is not None
                    and t.primary_key is None
                    and auto_detected_primary_key not in foreign_keys
                ):
                    t.primary_key = auto_detected_primary_key
                    rich.print(f"Detected for Table `{t.name}` primary key `{auto_detected_primary_key}`")
            else:
                for col in t.columns:
                    if col.model_encoding_type == ModelEncodingType.auto:
                        col.model_encoding_type = auto_detected_columns[col.name]

    # create generator
    # NOTE: model configurations will be revalidated by SourceTable
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
                        model_label=f"{table.name}:{model_type.value.lower()}",
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
