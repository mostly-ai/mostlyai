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

import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

from filelock import FileLock
from pydantic import BaseModel

from mostlyai.sdk.domain import (
    Connector,
    Dataset,
    Generator,
    JobProgress,
    ModelType,
    SourceTable,
    SyntheticDataset,
    SyntheticTable,
)


def get_model_label(
    table: str | SourceTable | SyntheticTable, model_type: str | ModelType, path_safe: bool = False
) -> str:
    """
    The model label is the name of the table with the model type. It is used to identify the model uniquely within the file storage as well as in the job progress.

    For usage in file storage, we need to ensure it's path safe. Therefore, we adapt the default infix from ":" to "=" on Windows environments.
    """
    table_name = table.name if hasattr(table, "name") else str(table)
    path_infix = "=" if path_safe and os.name == "nt" else ":"
    model_type = model_type.name if hasattr(model_type, "name") else ModelType(str(model_type).upper()).name
    return f"{table_name}{path_infix}{model_type}"


def convert_model_label_path(file_path: str) -> str:
    """
    Convert the model label to a OS-compatible valid file name.
    """
    if os.name == "nt":
        before, after = ":", "="
    else:
        before, after = "=", ":"
    if before in file_path:
        for model_type in ModelType:
            file_path = file_path.replace(f"{before}{model_type.name}", f"{after}{model_type.name}")
    return file_path


def read_generator_from_json(generator_dir: Path) -> Generator:
    json_file = generator_dir / "generator.json"
    return Generator(**json.loads(json_file.read_text()))


def write_generator_to_json(generator_dir: Path, generator: Generator) -> None:
    json_file = generator_dir / "generator.json"
    generator_dir.mkdir(parents=True, exist_ok=True)
    write_to_json(json_file, generator)


def read_connector_from_json(connector_dir: Path) -> Connector:
    json_file = connector_dir / "connector.json"
    return Connector(**json.loads(json_file.read_text()))


def write_connector_to_json(connector_dir: Path, connector: Connector) -> None:
    json_file = connector_dir / "connector.json"
    connector_dir.mkdir(parents=True, exist_ok=True)
    write_to_json(json_file, connector)


def read_dataset_from_json(dataset_dir: Path) -> Dataset:
    json_file = dataset_dir / "dataset.json"
    return Dataset(**json.loads(json_file.read_text()))


def write_dataset_to_json(dataset_dir: Path, dataset: Dataset) -> None:
    json_file = dataset_dir / "dataset.json"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_to_json(json_file, dataset)


def read_job_progress_from_json(resource_dir: Path) -> JobProgress:
    progress_file = resource_dir / "job_progress.json"
    lock_file = progress_file.with_suffix(".lock")
    lock = FileLock(lock_file)
    with lock:
        return JobProgress(**json.loads(progress_file.read_text()))


def write_job_progress_to_json(resource_dir: Path, job_progress: JobProgress) -> None:
    progress_file = resource_dir / "job_progress.json"
    resource_dir.mkdir(parents=True, exist_ok=True)
    write_to_json(progress_file, job_progress)


def read_synthetic_dataset_from_json(synthetic_dataset_dir: Path) -> SyntheticDataset:
    json_file = synthetic_dataset_dir / "synthetic-dataset.json"
    return SyntheticDataset(**json.loads(json_file.read_text()))


def write_synthetic_dataset_to_json(synthetic_dataset_dir: Path, synthetic_dataset: SyntheticDataset) -> None:
    json_file = synthetic_dataset_dir / "synthetic-dataset.json"
    synthetic_dataset_dir.mkdir(parents=True, exist_ok=True)
    write_to_json(json_file, synthetic_dataset)


def create_zip_in_memory(path: Path, pattern: str = "*") -> BytesIO:
    # Create an in-memory zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        if path.is_dir():
            # Iterate over all files in the directory and subdirectories
            for file_path in path.rglob(pattern):
                if file_path.is_file():  # Ensure it's a file (not a directory)
                    # Add file to the zip archive, keeping the directory structure
                    zip_file.write(file_path, file_path.relative_to(path))
        else:
            # Add the single file to the zip archive
            zip_file.write(path, path.name)
    # Ensure the buffer is ready for streaming
    zip_buffer.seek(0)
    return zip_buffer


def write_to_json(file_path: Path, obj: BaseModel) -> None:
    json_str = obj.model_dump_json(
        # pretty print JSON
        indent=2,
        # serialize with camelCase aliases for platform compatibility
        by_alias=True,
    )
    lock_file = file_path.with_suffix(".lock")
    lock = FileLock(lock_file)
    with lock:
        file_path.write_text(json_str)
