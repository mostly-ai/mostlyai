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
import os
import shutil
import subprocess
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse

from mostlyai import sdk
from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._local import generators, synthetic_datasets
from mostlyai.sdk._local.execution.jobs import execute_probing_job
from mostlyai.sdk._local.generators import create_generator as create_generator_model
from mostlyai.sdk._local import connectors
from mostlyai.sdk.domain import (
    AboutService,
    CurrentUser,
    Generator,
    GeneratorPatchConfig,
    GeneratorListItem,
    GeneratorConfig,
    ProgressStatus,
    JobProgress,
    ConnectorConfig,
    Connector,
    ConnectorListItem,
    SyntheticDataset,
    SyntheticDatasetConfig,
    SyntheticDatasetPatchConfig,
    SyntheticDatasetFormat,
    ConnectorPatchConfig,
    ComputeListItem,
    ModelType,
    SyntheticProbeConfig,
    Probe,
    GeneratorCloneConfig,
    GeneratorCloneTrainingStatus,
)
from mostlyai.sdk._local.storage import (
    read_generator_from_json,
    write_generator_to_json,
    read_job_progress_from_json,
    read_connector_from_json,
    read_synthetic_dataset_from_json,
    write_synthetic_dataset_to_json,
    create_zip_in_memory,
    write_connector_to_json,
)
from mostlyai.sdk._data.file.utils import read_data_table_from_path


class Routes:
    def __init__(self, home_dir: Path):
        self.home_dir = home_dir
        self.router = APIRouter()
        self._initialize_routes()

    def _html(self, title: str, body: str):
        return f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>{ title }</title>
                    <style>
                        body {{
                            font-family: monospace;
                        }}
                        pre {{
                            background: #f4f4f4;
                            padding: 10px;
                            border-radius: 5px;
                            overflow-x: auto;
                        }}
                    </style>
                </head>
                <body>
                    <h1>{ title }</h1>
                    { body }
                </body>
                </html>
                """

    def _initialize_routes(self):
        ## GENERAL

        @self.router.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/docs")

        @self.router.get("/d/connectors/{id}", response_class=HTMLResponse, include_in_schema=False)
        async def show_connector(id: str) -> HTMLResponse:
            connector_dir = self.home_dir / "connectors" / id
            connector_json = connector_dir / "connector.json"
            body = f"<h3>file://{connector_json}</h3>" f"<pre>{connector_json.read_text()}</pre>"
            return HTMLResponse(content=self._html("Connector", body))

        @self.router.get("/d/generators/{id}", response_class=HTMLResponse, include_in_schema=False)
        async def show_generator(id: str) -> HTMLResponse:
            generator_dir = self.home_dir / "generators" / id
            generator_json = generator_dir / "generator.json"
            progress_json = generator_dir / "job_progress.json"
            job_log_file = generator_dir / "job.log"
            job_log_logs = job_log_file.read_text() if job_log_file.exists() else ""
            body = (
                f"<h3>file://{generator_json}</h3>"
                f"<pre>{generator_json.read_text()}</pre>"
                f"<h3>file://{progress_json}</h3>"
                f"<pre>{progress_json.read_text()}</pre>"
                f"<h3>file://{job_log_file}</h3>"
                f"<pre>{job_log_logs}</pre>"
            )
            return HTMLResponse(content=self._html("Generator", body))

        @self.router.get("/d/synthetic-datasets/{id}", response_class=HTMLResponse, include_in_schema=False)
        async def show_synthetic_dataset(id: str) -> HTMLResponse:
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            synthetic_dataset_json = synthetic_dataset_dir / "synthetic-dataset.json"
            progress_json = synthetic_dataset_dir / "job_progress.json"
            job_log_file = synthetic_dataset_dir / "job.log"
            job_log_logs = job_log_file.read_text() if job_log_file.exists() else ""
            body = (
                f"<h3>file://{synthetic_dataset_json}</h3>"
                f"<pre>{synthetic_dataset_json.read_text()}</pre>"
                f"<h3>file://{progress_json}</h3>"
                f"<pre>{progress_json.read_text()}</pre>"
                f"<h3>file://{job_log_file}</h3>"
                f"<pre>{job_log_logs}</pre>"
            )
            return HTMLResponse(content=self._html("Synthetic Dataset", body))

        @self.router.get("/about", response_model=AboutService)
        async def get_about_service() -> AboutService:
            return AboutService(version=sdk.__version__, assistant=False)

        @self.router.get("/users/me", response_model=CurrentUser)
        async def get_current_user_info() -> CurrentUser:
            return CurrentUser()

        @self.router.get("/computes", response_model=list[ComputeListItem])
        async def list_computes() -> JSONResponse:
            return JSONResponse(content=[])

        @self.router.get("/models/{model_type}")
        async def list_models(model_type: str) -> JSONResponse:
            if model_type == ModelType.tabular:
                return JSONResponse(content=["MOSTLY_AI/Small", "MOSTLY_AI/Medium", "MOSTLY_AI/Large"])
            elif model_type == ModelType.language:
                return JSONResponse(
                    content=["MOSTLY_AI/LSTMFromScratch-3m", "microsoft/phi-1_5", "(HuggingFace-hosted models)"]
                )
            return JSONResponse(content=[])

        ## CONNECTORS

        @self.router.get("/connectors")
        async def list_connectors(
            offset=0, limit=50, searchTerm: str | None = None, access_type: str | None = None
        ) -> JSONResponse:
            connector_dirs = [p for p in (self.home_dir / "connectors").glob("*") if p.is_dir()]
            connector_list_items = []
            for connector_dir in connector_dirs:
                connector = read_connector_from_json(connector_dir)
                if not searchTerm or searchTerm.lower() in (connector.name or "").lower():
                    if not access_type or access_type == connector.access_type:
                        connector_list_items.append(ConnectorListItem(**connector.model_dump()))

            # use jsonable_encoder to handle non-serializable objects like datetime
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(
                    {
                        "totalCount": len(connector_list_items),
                        "results": connector_list_items[int(offset) : int(offset) + int(limit)],
                    }
                ),
            )

        @self.router.post("/connectors", response_model=Connector)
        async def create_connector(config: ConnectorConfig = Body(...), testConnection: bool = True) -> Connector:
            connector = connectors.create_connector(self.home_dir, config, test_connection=testConnection)
            return connector

        @self.router.get("/connectors/{id}", response_model=Connector)
        async def get_connector(id: str) -> Connector:
            connector_dir = self.home_dir / "connectors" / id
            if not connector_dir.exists():
                raise HTTPException(status_code=404, detail=f"Connector `{id}` not found")
            connector = read_connector_from_json(connector_dir)
            return connector

        @self.router.patch("/connectors/{id}", response_model=Connector)
        async def patch_connector(
            id: str, config: ConnectorPatchConfig = Body(...), testConnection: bool = True
        ) -> Connector:
            config = connectors.encrypt_connector_config(config)
            connector_dir = self.home_dir / "connectors" / id
            connector = read_connector_from_json(connector_dir)
            for key, value in config.model_dump().items():
                if value is not None:
                    setattr(connector, key, value)
            if testConnection:
                connectors.do_test_connection(connector)
            write_connector_to_json(connector_dir, connector)
            return connector

        @self.router.delete("/connectors/{id}")
        async def delete_connector(id: str):
            connector_dir = self.home_dir / "connectors" / id
            shutil.rmtree(connector_dir, ignore_errors=True)

        @self.router.get("/connectors/{id}/locations")
        async def list_connector_locations(id: str, prefix: str):
            connector_dir = self.home_dir / "connectors" / id
            connector = read_connector_from_json(connector_dir)
            container = create_container_from_connector(connector)
            locations = container.list_locations(prefix)
            return locations

        @self.router.get("/connectors/{id}/schema")
        async def location_schema(id: str, location: str):
            connector_dir = self.home_dir / "connectors" / id
            connector = read_connector_from_json(connector_dir)
            container = create_container_from_connector(connector)
            meta = container.set_location(location)
            if hasattr(container, "dbname"):
                table_name = meta["table_name"]
                container.filtered_tables = [table_name]
                container.fetch_schema()
                table = container.schema.tables.get(table_name)
            else:
                table = read_data_table_from_path(container)
            if not table:
                return JSONResponse(status_code=400, content=f"Table at {location} not found")
            columns = [
                {
                    "name": col,
                    "originalDataType": str(table.dtypes[col].wrapped),
                    "defaultModelEncodingType": table.encoding_types[col].value,
                }
                for col in table.columns
            ]
            return JSONResponse(status_code=200, content=columns)

        ## GENERATORS

        @self.router.get("/generators")
        async def list_generators(
            offset: int = 0, limit: int = 50, searchTerm: str | None = None, status: str | None = None
        ) -> JSONResponse:
            generator_dirs = [p for p in (self.home_dir / "generators").glob("*") if p.is_dir()]
            generator_list_items = []
            for generator_dir in generator_dirs:
                generator = read_generator_from_json(generator_dir)
                if (
                    not searchTerm
                    or searchTerm.lower() in (generator.name or "").lower()
                    or searchTerm.lower() in (generator.description or "").lower()
                ):
                    if not status or status == generator.training_status:
                        generator_list_items.append(GeneratorListItem(**generator.model_dump()))
            # use jsonable_encoder to handle non-serializable objects like datetime
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(
                    {
                        "totalCount": len(generator_list_items),
                        "results": generator_list_items[int(offset) : int(offset) + int(limit)],
                    }
                ),
            )

        @self.router.post("/generators", response_model=Generator)
        async def create_generator(config: GeneratorConfig = Body(...)) -> Generator:
            generator = generators.create_generator(self.home_dir, config)
            return generator

        @self.router.get("/generators/{id}", response_model=Generator)
        async def get_generator(id: str) -> Generator:
            generator_dir = self.home_dir / "generators" / id
            if not generator_dir.exists():
                raise HTTPException(status_code=404, detail=f"Generator `{id}` not found")
            generator = read_generator_from_json(generator_dir)
            return generator

        @self.router.patch("/generators/{id}", response_model=Generator)
        async def patch_generator(id: str, config: GeneratorPatchConfig = Body(...)) -> Generator:
            generator_dir = self.home_dir / "generators" / id
            generator = read_generator_from_json(generator_dir)
            for key, value in config.model_dump().items():
                if value is not None:
                    setattr(generator, key, value)
            write_generator_to_json(generator_dir, generator)
            return generator

        @self.router.delete("/generators/{id}")
        async def delete_generator(id: str):
            generator_dir = self.home_dir / "generators" / id
            shutil.rmtree(generator_dir, ignore_errors=True)

        @self.router.post("/generators/{id}/clone", response_model=Generator)
        async def clone_generator(id: str, config: GeneratorCloneConfig = Body(...)) -> Generator:
            generator_dir = self.home_dir / "generators" / id
            generator = read_generator_from_json(generator_dir)
            new_generator = create_generator_model(
                home_dir=self.home_dir, config=GeneratorConfig(**generator.model_dump())
            )
            new_generator_dir = self.home_dir / "generators" / new_generator.id
            if config.training_status == GeneratorCloneTrainingStatus.continue_:
                new_generator.training_status = ProgressStatus.continue_
                shutil.copytree(generator_dir / "ModelStore", new_generator_dir / "ModelStore")
            return new_generator

        @self.router.get("/generators/{id}/tables/{table_id}/report", response_class=HTMLResponse)
        async def get_model_report(id: str, table_id: str, modelType: str) -> HTMLResponse:
            generator_dir = self.home_dir / "generators" / id
            generator = read_generator_from_json(generator_dir)
            table = next((t for t in generator.tables if t.id == table_id), None)
            reports_dir = self.home_dir / "generators" / id / "ModelQAReports"
            fn = reports_dir / f"{table.name}:{modelType.lower()}.html"
            return HTMLResponse(content=fn.read_text())

        @self.router.get("/generators/{id}/training", response_model=JobProgress)
        async def get_training_progress(id: str) -> JobProgress:
            generator_dir = self.home_dir / "generators" / id
            job_progress = read_job_progress_from_json(generator_dir)
            return job_progress

        @self.router.post("/generators/{id}/training/start", response_model=None)
        async def start_training(id: str):
            generator_dir = self.home_dir / "generators" / id
            generator = read_generator_from_json(generator_dir)

            # check valid training_status
            if generator.training_status not in [ProgressStatus.new, ProgressStatus.continue_]:
                raise HTTPException(status_code=400)

            # call shell script to start training job
            cli_py = str((Path(os.path.dirname(os.path.realpath(__file__))) / "cli.py").absolute())
            cmd = ["python", cli_py, "run-training", generator.id, str(self.home_dir.absolute())]
            subprocess.Popen(cmd)

        @self.router.get("/generators/{id}/export-to-file", response_class=StreamingResponse)
        async def export_generator_to_file(id: str) -> StreamingResponse:
            generator_dir = self.home_dir / "generators" / id
            zip_buffer = create_zip_in_memory(generator_dir)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=generator-{id[:8]}.zip"},
            )

        @self.router.post("/generators/import-from-file", response_model=Generator)
        async def import_generator_from_file(file: UploadFile = File(...)) -> Generator:
            generator_id = str(uuid.uuid4())  # generate new UUID
            generator_dir = self.home_dir / "generators" / generator_id
            generator_dir.mkdir(parents=True, exist_ok=True)

            file_content = await file.read()
            try:
                with zipfile.ZipFile(BytesIO(file_content)) as zip_ref:
                    zip_ref.extractall(generator_dir)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file")

            generator = read_generator_from_json(generator_dir)
            generator.id = generator_id
            for tables in generator.tables:
                tables.source_connector_id = None
            generator.metadata = None
            write_generator_to_json(generator_dir, generator)
            return generator

        @self.router.get("/generators/{id}/config", response_model=GeneratorConfig)
        async def get_generator_config(id: str) -> GeneratorConfig:
            generator_dir = self.home_dir / "generators" / id
            generator = read_generator_from_json(generator_dir)
            config = GeneratorConfig(**generator.model_dump())
            return config

        ### SYNTHETIC DATASETS

        @self.router.get("/synthetic-datasets")
        async def list_synthetic_datasets(
            offset: int = 0, limit: int = 50, searchTerm: str | None = None, status: str | None = None
        ) -> JSONResponse:
            synthetic_dataset_dirs = [p for p in (self.home_dir / "synthetic-datasets").glob("*") if p.is_dir()]
            synthetic_dataset_list_items = []
            for synthetic_dataset_dir in synthetic_dataset_dirs:
                synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
                if (
                    not searchTerm
                    or searchTerm.lower() in (synthetic_dataset.name or "").lower()
                    or searchTerm.lower() in (synthetic_dataset.description or "").lower()
                ):
                    if not status or status == synthetic_dataset.generation_status:
                        synthetic_dataset_list_items.append(SyntheticDataset(**synthetic_dataset.model_dump()))
            # use jsonable_encoder to handle non-serializable objects like datetime
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(
                    {
                        "totalCount": len(synthetic_dataset_list_items),
                        "results": synthetic_dataset_list_items[int(offset) : int(offset) + int(limit)],
                    }
                ),
            )

        @self.router.post("/synthetic-datasets", response_model=SyntheticDataset)
        async def create_synthetic_dataset(config: SyntheticDatasetConfig = Body(...)) -> SyntheticDataset:
            synthetic_dataset = synthetic_datasets.create_synthetic_dataset(self.home_dir, config)
            return synthetic_dataset

        @self.router.get("/synthetic-datasets/{id}", response_model=SyntheticDataset)
        async def get_synthetic_dataset(id: str) -> SyntheticDataset:
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            if not synthetic_dataset_dir.exists():
                raise HTTPException(status_code=404, detail=f"Synthetic Dataset `{id}` not found")
            synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
            return synthetic_dataset

        @self.router.patch("/synthetic-datasets/{id}", response_model=SyntheticDataset)
        async def patch_synthetic_dataset(id: str, config: SyntheticDatasetPatchConfig = Body(...)) -> SyntheticDataset:
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
            for key, value in config.model_dump().items():
                if value is not None:
                    setattr(synthetic_dataset, key, value)
            write_synthetic_dataset_to_json(synthetic_dataset_dir, synthetic_dataset)
            return synthetic_dataset

        @self.router.delete("/synthetic-datasets/{id}")
        async def delete_synthetic_dataset(id: str):
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            shutil.rmtree(synthetic_dataset_dir, ignore_errors=True)

        @self.router.get("/generators/{id}/tables/{table_id}/report", response_class=HTMLResponse)
        async def get_data_report(id: str, table_id: str, reportType: str, modelType: str) -> HTMLResponse:
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
            table = next((t for t in synthetic_dataset.tables if t.id == table_id), None)
            if reportType == "MODEL":
                reports_dir = self.home_dir / "synthetic-datasets" / id / "ModelQAReports"
            else:
                reports_dir = self.home_dir / "synthetic-datasets" / id / "DataQAReports"
            fn = reports_dir / f"{table.name}:{reportType.lower()}.html"
            return HTMLResponse(content=fn.read_text())

        @self.router.get("/synthetic-datasets/{id}/generation", response_model=JobProgress)
        async def get_generation_progress(id: str) -> JobProgress:
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            job_progress = read_job_progress_from_json(synthetic_dataset_dir)
            return job_progress

        @self.router.post("/synthetic-datasets/{id}/generation/start", response_model=None)
        async def start_generation(id: str):
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)

            # check valid generation_status
            if synthetic_dataset.generation_status not in [ProgressStatus.new, ProgressStatus.continue_]:
                raise HTTPException(status_code=400)

            # call shell script to start generation job
            cli_py = str((Path(os.path.dirname(os.path.realpath(__file__))) / "cli.py").absolute())
            cmd = ["python", cli_py, "run-generation", synthetic_dataset.id, str(self.home_dir.absolute())]
            subprocess.Popen(cmd)

        @self.router.get("/synthetic-datasets/{id}/config", response_model=SyntheticDatasetConfig)
        async def get_synthetic_dataset_config(id: str) -> SyntheticDatasetConfig:
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
            config = SyntheticDatasetConfig(**synthetic_dataset.model_dump())
            return config

        @self.router.get("/synthetic-datasets/{id}/download", response_class=FileResponse)
        async def download_synthetic_dataset(id: str, slft: str, format: str) -> FileResponse:
            _ = slft  # ignore parameter
            synthetic_dataset_dir = self.home_dir / "synthetic-datasets" / id
            if format == SyntheticDatasetFormat.parquet:
                filename = "synthetic-parquet-data.zip"
            elif format == SyntheticDatasetFormat.csv:
                filename = "synthetic-csv-data.zip"
            elif format == SyntheticDatasetFormat.xlsx:
                filename = "synthetic-samples.xlsx"
            else:
                raise HTTPException(status_code=400, detail="Invalid format")

            return FileResponse(
                path=(synthetic_dataset_dir / "ZIP" / filename).absolute(),
            )

        ### SYNTHETIC PROBES

        @self.router.post("/synthetic-probes", response_model=list[Probe])
        async def create_synthetic_probe(config: SyntheticProbeConfig = Body(...)) -> list[Probe]:
            synthetic_dataset = synthetic_datasets.create_synthetic_dataset(home_dir=self.home_dir, config=config)
            return execute_probing_job(synthetic_dataset_id=synthetic_dataset.id, home_dir=self.home_dir)