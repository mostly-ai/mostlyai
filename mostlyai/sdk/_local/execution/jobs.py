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
import logging
import shutil
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd

from mostlyai import engine
from mostlyai.sdk._data.file.base import LocalFileContainer
from mostlyai.sdk._data.file.utils import make_data_table_from_container
from mostlyai.sdk._data.util.common import TABLE_COLUMN_INFIX, TEMPORARY_PRIMARY_KEY, strip_column_prefix
from mostlyai.sdk._local.execution.plan import (
    ExecutionPlan,
    Task,
    make_generator_execution_plan,
    make_synthetic_dataset_execution_plan,
)
from mostlyai.sdk._local.execution.step_analyze_training_data import execute_step_analyze_training_data
from mostlyai.sdk._local.execution.step_create_data_report import execute_step_create_data_report
from mostlyai.sdk._local.execution.step_create_model_report import (
    execute_step_create_model_report,
)
from mostlyai.sdk._local.execution.step_deliver_data import execute_step_deliver_data
from mostlyai.sdk._local.execution.step_encode_training_data import execute_step_encode_training_data
from mostlyai.sdk._local.execution.step_finalize_generation import (
    create_generation_schema,
    execute_step_finalize_generation,
    update_total_rows,
)
from mostlyai.sdk._local.execution.step_generate_data import execute_step_generate_data
from mostlyai.sdk._local.execution.step_generate_model_report_data import (
    execute_step_generate_model_report_data,
)
from mostlyai.sdk._local.execution.step_pull_training_data import (
    execute_step_pull_training_data,
)
from mostlyai.sdk._local.execution.step_train_model import execute_step_train_model
from mostlyai.sdk._local.progress import LocalProgressCallback, get_current_utc_time
from mostlyai.sdk._local.storage import (
    get_model_label,
    read_connector_from_json,
    read_generator_from_json,
    read_job_progress_from_json,
    read_synthetic_dataset_from_json,
    write_generator_to_json,
    write_job_progress_to_json,
    write_synthetic_dataset_to_json,
)
from mostlyai.sdk._local.synthetic_datasets import create_synthetic_dataset
from mostlyai.sdk.domain import (
    ConnectorType,
    Generator,
    ModelType,
    Probe,
    ProgressStatus,
    SourceColumn,
    StepCode,
    SyntheticDataset,
    SyntheticProbeConfig,
    TaskType,
)

_LOG = logging.getLogger(__name__)


def _set_random_state(random_state: int | None = None):
    # rely on engine to set global random state
    engine.set_random_state(random_state)


def _move_training_artefacts(generator_dir: Path, job_workspace_dir: Path):
    for dir in ["Logs", "ModelStore", "ModelQAReports", "ModelQAStatistics"]:
        shutil.rmtree(generator_dir / dir, ignore_errors=True)
        (generator_dir / dir).mkdir()
    for path in job_workspace_dir.absolute().rglob("*"):
        if path.is_dir() and path.name == "ModelStore":
            model_label = path.parent.name
            path.rename(generator_dir / "ModelStore" / model_label)
        if path.is_file() and path.parent.name == "ModelQAReports":
            path.rename(generator_dir / "ModelQAReports" / path.name)
        if path.is_dir() and path.name == "ModelQAStatistics":
            model_label = path.parent.name
            path.rename(generator_dir / "ModelQAStatistics" / model_label)


def _move_generation_artefacts(synthetic_dataset_dir: Path, job_workspace_dir: Path):
    for path in job_workspace_dir.absolute().rglob("*"):
        target = None
        if path.is_dir() and path.name == "RandomSamples":
            target = synthetic_dataset_dir / "RandomSamples"
        elif path.is_dir() and path.name == "ZIP":
            target = synthetic_dataset_dir / "ZIP"
        elif path.is_file() and path.parent.name == "DataQAReports":
            target = synthetic_dataset_dir / "DataQAReports" / path.name

        if target:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))


def _mark_in_progress(resource: Generator | SyntheticDataset, resource_dir: Path):
    if isinstance(resource, Generator):
        resource.training_status = ProgressStatus.in_progress
    else:
        resource.generation_status = ProgressStatus.in_progress
    job_progress = read_job_progress_from_json(resource_dir)
    job_progress.status = ProgressStatus.in_progress
    for step in job_progress.steps:
        step.status = ProgressStatus.in_progress
    write_job_progress_to_json(resource_dir, job_progress)


def _mark_done(resource: Generator | SyntheticDataset, resource_dir: Path):
    if isinstance(resource, Generator):
        resource.training_status = ProgressStatus.done
    else:
        resource.generation_status = ProgressStatus.done
    now = get_current_utc_time()
    job_progress = read_job_progress_from_json(resource_dir)
    job_progress.status = ProgressStatus.done
    job_progress.start_date = job_progress.start_date or now
    job_progress.end_date = job_progress.end_date or now
    job_progress.progress.value = job_progress.progress.max
    for step in job_progress.steps:
        step.status = ProgressStatus.done
        step.start_date = step.start_date or now
        step.end_date = step.end_date or now
        step.progress.value = step.progress.max
    write_job_progress_to_json(resource_dir, job_progress)


def _mark_failed(resource: Generator | SyntheticDataset, resource_dir: Path):
    if isinstance(resource, Generator):
        resource.training_status = ProgressStatus.failed
    else:
        resource.generation_status = ProgressStatus.failed
    job_progress = read_job_progress_from_json(resource_dir)
    job_progress.status = ProgressStatus.failed
    for step in job_progress.steps:
        if step.status != ProgressStatus.done:
            step.status = ProgressStatus.failed
    write_job_progress_to_json(resource_dir, job_progress)


def _copy_model(generator_dir: Path, model_label: str, workspace_dir: Path) -> bool:
    model_path = generator_dir / "ModelStore" / model_label
    if model_path.exists():
        shutil.copytree(model_path, workspace_dir / "ModelStore")
        return True
    return False


def _copy_statistics(generator_dir: Path, model_label: str, workspace_dir: Path) -> bool:
    statistics_path = generator_dir / "ModelQAStatistics" / model_label
    if statistics_path.exists():
        shutil.copytree(statistics_path, workspace_dir / "ModelQAStatistics")
        return True
    return False


def _set_overall_accuracy(generator: Generator) -> None:
    accuracy = [
        metrics.accuracy.overall
        for table in generator.tables
        for metrics in [table.tabular_model_metrics, table.language_model_metrics]
        if metrics is not None and metrics.accuracy.overall is not None
    ]
    generator.accuracy = sum(accuracy) / len(accuracy) if accuracy else None


def _probe_random_samples(home_dir: Path, generator: Generator) -> None:
    config = SyntheticProbeConfig(generator_id=generator.id)
    synthetic_dataset = create_synthetic_dataset(home_dir=home_dir, config=config, sample_size=20)
    probes = execute_probing_job(synthetic_dataset_id=synthetic_dataset.id, home_dir=home_dir)

    random_samples_dir = home_dir / "generators" / generator.id / "RandomSamples"
    random_samples_dir.mkdir(parents=True, exist_ok=True)
    for probe in probes:
        df = pd.DataFrame(probe.rows)
        df.to_json(random_samples_dir / f"{probe.name}.json", orient="records")


def _merge_tabular_language_data(workspace_dir: Path, merged_part_size: int = 200_000):
    def rows_from_dir_generator(path: Path, yield_n_rows: int):
        fns = sorted(list(path.glob("*.parquet")))
        buffer = pd.DataFrame()
        for fn in fns:
            fn_df = pd.read_parquet(fn)
            buffer = pd.concat([buffer, fn_df], ignore_index=True)
            while len(buffer) >= yield_n_rows:
                yield buffer.iloc[:yield_n_rows]
                buffer = buffer.iloc[yield_n_rows:].reset_index(drop=True)
        if not buffer.empty:
            yield buffer

    ctx_dir = workspace_dir / "OriginalData" / "ctx-data"
    tgt_dir = workspace_dir / "SyntheticData"
    merge_dir = workspace_dir / "_TEMP_MERGE_DIR_"
    ctx_gen = rows_from_dir_generator(ctx_dir, yield_n_rows=merged_part_size)
    tgt_gen = rows_from_dir_generator(tgt_dir, yield_n_rows=merged_part_size)
    merge_dir.mkdir(parents=True, exist_ok=True)
    for idx, ctx_df in enumerate(ctx_gen):
        # assumption: rows from ctx_dir form 1:1 mapping with rows from tgt_dir
        tgt_df = next(tgt_gen)
        # assumption: all columns are prefixed
        tgt_prefix = [col.split(TABLE_COLUMN_INFIX)[0] for col in ctx_df.columns if TEMPORARY_PRIMARY_KEY in col][0]
        # only merge TABULAR (which are in context) and LANGUAGE columns
        ctx_df = ctx_df.drop(columns=[c for c in ctx_df.columns if not c.startswith(tgt_prefix)])
        ctx_df.columns = strip_column_prefix(ctx_df.columns, table_name=tgt_prefix)
        df = pd.concat([ctx_df, tgt_df], axis=1)
        # remove temporary primary key
        df = df.loc[:, ~df.columns.str.contains(TEMPORARY_PRIMARY_KEY)]
        out_fn = merge_dir / f"part.{idx:06}.{0:06}.parquet"
        df.to_parquet(out_fn)
    assert next(tgt_gen, None) is None

    # re-create generated data dir with merged data
    shutil.rmtree(workspace_dir / "SyntheticData", ignore_errors=True)
    shutil.move(merge_dir, workspace_dir / "SyntheticData")


def _fetch_sample_seed(home_dir: Path, connector_id: str) -> pd.DataFrame:
    # create seed data container
    connector_id = str(connector_id)
    data_container = LocalFileContainer()
    location = str((home_dir / "connectors" / connector_id / "seed.parquet").absolute())
    data_container.set_location(location)

    # create seed data table
    data_table = make_data_table_from_container(data_container)
    data_table.is_output = False
    return data_table.read_data()


### PLAN EXECUTION ###


class Execution:
    def __init__(
        self,
        *,
        execution_plan: ExecutionPlan,
        generator: Generator,
        synthetic_dataset: SyntheticDataset | None = None,
        home_dir: Path,
    ):
        self._execution_plan = execution_plan
        self._generator = generator
        self._synthetic_dataset = synthetic_dataset
        self._home_dir = home_dir
        if synthetic_dataset is None:
            self._resource_dir = self._home_dir / "generators" / generator.id
            self._job_workspace_dir = self._home_dir / "in_progress" / generator.id
        else:
            self._resource_dir = self._home_dir / "synthetic-datasets" / synthetic_dataset.id
            self._job_workspace_dir = self._home_dir / "in_progress" / synthetic_dataset.id
        self._job_workspace_dir.mkdir(parents=True)
        # set up logging
        logging.basicConfig(
            filename=(self._resource_dir / "job.log").absolute(),
            filemode="w",
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-7s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def clear_job_workspace(self):
        shutil.rmtree(self._job_workspace_dir, ignore_errors=True)

    def clear_file_upload_connectors(self):
        connector_ids = []
        if self._generator is not None:
            connector_ids += [t.source_connector_id for t in self._generator.tables]
        if self._synthetic_dataset is not None:
            connector_ids += [
                t.configuration.sample_seed_connector_id
                for t in self._synthetic_dataset.tables
                if t.configuration.sample_seed_connector_id is not None
            ]
        connectors_dir = self._home_dir / "connectors"
        if connectors_dir.exists():
            for connector_path in connectors_dir.iterdir():
                if not (connector_path.name in connector_ids and connector_path.is_dir()):
                    continue

                try:
                    connector = read_connector_from_json(connector_path)
                    if connector.type == ConnectorType.file_upload:
                        shutil.rmtree(connector_path)
                except Exception as e:
                    _LOG.info(f"Failed to clear connector {connector_path}: {e}")
                    pass

    def run(self):
        # execute job
        task_handlers: dict[TaskType, Callable] = {
            TaskType.sync: self.execute_task_sync,
            TaskType.train_tabular: self.execute_task_train,
            TaskType.train_language: self.execute_task_train,
            TaskType.finalize_training: self.execute_task_finalize_training,
            TaskType.generate: self.execute_task_generate,
            TaskType.probe: self.execute_task_generate,
        }
        for task in self._execution_plan.tasks:
            handler = task_handlers.get(task.type)
            if handler is None:
                raise ValueError(f"invalid task type: {task.type}")
            handler(task)

    def execute_task_sync(self, task: Task):
        pass

    def execute_task_train(self, task: Task):
        # gather arguments that are common across steps
        generator = self._generator
        tgt_table = next(t for t in generator.tables if t.name == task.target_table_name)
        model_type = ModelType.tabular if task.type == TaskType.train_tabular else ModelType.language
        model_label = get_model_label(tgt_table, model_type, path_safe=False)
        model_label_path = get_model_label(tgt_table, model_type, path_safe=True)
        generator_dir = self._home_dir / "generators" / generator.id
        workspace_dir = self._job_workspace_dir / model_label_path
        workspace_dir.mkdir(parents=True, exist_ok=True)
        update_progress_fn = partial(LocalProgressCallback, resource_path=generator_dir, model_label=model_label)

        # let's first copy the ModelStore in case that it already exists (training continuation)
        _copy_model(generator_dir=generator_dir, model_label=model_label_path, workspace_dir=workspace_dir)

        # step: PULL_TRAINING_DATA
        connectors = [
            read_connector_from_json(self._home_dir / "connectors" / t.source_connector_id) for t in generator.tables
        ]
        columns, total_rows = execute_step_pull_training_data(
            generator=generator,
            connectors=connectors,
            model_type=model_type,
            target_table_name=task.target_table_name,
            workspace_dir=workspace_dir,
            update_progress=update_progress_fn(step_code=StepCode.pull_training_data),
        )
        # update generator with columns, in case they haven't been set yet (allows to run jobs without column info)
        if tgt_table.columns is None:
            tgt_table.columns = [SourceColumn(name=col) for col in columns]
        # update generator with table total rows, so that value can be used for default sample size during generation
        tgt_table.total_rows = total_rows

        # step: ANALYZE_TRAINING_DATA
        encoding_types, value_ranges = execute_step_analyze_training_data(
            generator=generator,
            model_type=model_type,
            target_table_name=task.target_table_name,
            workspace_dir=workspace_dir,
            update_progress=update_progress_fn(step_code=StepCode.analyze_training_data),
        )
        # update generator with final encoding types, so that it is known how AUTO got resolved
        for column in tgt_table.columns:
            if column.name in encoding_types:
                column.model_encoding_type = encoding_types[column.name]
        # update generator with column-level value ranges, so that these are known for conditional generation
        for column in tgt_table.columns:
            if column.name in value_ranges:
                column.value_range = value_ranges[column.name]

        # step: ENCODE_TRAINING_DATA
        execute_step_encode_training_data(
            workspace_dir=workspace_dir,
            update_progress=update_progress_fn(step_code=StepCode.encode_training_data),
        )

        # step: TRAIN_MODEL
        execute_step_train_model(
            generator=generator,
            model_type=model_type,
            target_table_name=task.target_table_name,
            restarts=0,  # "resume training" is not supported in local mode
            workspace_dir=workspace_dir,
            update_progress=update_progress_fn(step_code=StepCode.train_model),
            upload_model_data_callback=None,  # not applicable in local mode
        )

        step_codes = [s.step_code for s in task.steps]
        if StepCode.generate_model_report_data in step_codes:
            # step: GENERATE_MODEL_REPORT_DATA
            execute_step_generate_model_report_data(
                workspace_dir=workspace_dir,
                model_type=model_type,
                update_progress=update_progress_fn(step_code=StepCode.generate_model_report_data),
            )

            # step: CREATE_MODEL_REPORT
            model_metrics = execute_step_create_model_report(
                generator=generator,
                model_type=model_type,
                target_table_name=task.target_table_name,
                workspace_dir=workspace_dir,
                update_progress=update_progress_fn(step_code=StepCode.create_model_report),
            )
            # update generator with model metrics
            if model_type == ModelType.tabular:
                tgt_table.tabular_model_metrics = model_metrics
            else:
                tgt_table.language_model_metrics = model_metrics

    def execute_task_finalize_training(self, task: Task):
        pass

    def execute_task_generate(self, task: Task):
        generator = self._generator
        synthetic_dataset = self._synthetic_dataset
        synthetic_dataset_dir = self._home_dir / "synthetic-datasets" / synthetic_dataset.id
        generator_dir = self._home_dir / "generators" / generator.id

        visited_tables = set()
        table_lookup = {table.name: table for table in synthetic_dataset.tables}

        for step in task.steps:
            model_type = (
                ModelType.tabular
                if step.step_code in {StepCode.generate_data_tabular, StepCode.create_data_report_tabular}
                else ModelType.language
            )
            model_label = get_model_label(step.target_table_name, model_type, path_safe=False)
            model_label_path = get_model_label(step.target_table_name, model_type, path_safe=True)
            workspace_dir = self._job_workspace_dir / model_label_path
            workspace_dir.mkdir(exist_ok=True)

            update_progress = partial(
                LocalProgressCallback, resource_path=synthetic_dataset_dir, model_label=model_label
            )

            if step.step_code in {StepCode.generate_data_tabular, StepCode.generate_data_language}:
                # step: GENERATE_DATA
                _copy_model(generator_dir=generator_dir, model_label=model_label_path, workspace_dir=workspace_dir)

                visited_tables.add(step.target_table_name)

                table = table_lookup[step.target_table_name]
                sample_seed = (
                    _fetch_sample_seed(
                        home_dir=self._home_dir, connector_id=table.configuration.sample_seed_connector_id
                    )
                    if table.configuration.sample_seed_connector_id
                    else None
                )

                schema = create_generation_schema(
                    generator=generator,
                    job_workspace_dir=self._job_workspace_dir,
                    step="pull_context_data",
                )

                execute_step_generate_data(
                    generator=generator,
                    synthetic_dataset=synthetic_dataset,
                    target_table_name=step.target_table_name,
                    model_type=model_type,
                    sample_seed=sample_seed,
                    schema=schema,
                    workspace_dir=workspace_dir,
                    update_progress=update_progress(step_code=step.step_code),
                )

            elif step.step_code in {StepCode.create_data_report_tabular, StepCode.create_data_report_language}:
                model_report_available = _copy_statistics(
                    generator_dir=generator_dir, model_label=model_label_path, workspace_dir=workspace_dir
                )
                if not model_report_available:
                    continue
                # step: GENERATE_DATA_REPORT
                execute_step_create_data_report(
                    generator=generator,
                    target_table_name=step.target_table_name,
                    model_type=model_type,
                    workspace_dir=workspace_dir,
                    update_progress=update_progress(step_code=step.step_code),
                )

            elif step.step_code in {StepCode.finalize_generation, StepCode.finalize_probing}:
                # for every LANGUAGE model generation, merge context and generated data
                for table_name in visited_tables:
                    language_path = self._job_workspace_dir / get_model_label(
                        table_lookup[table_name], ModelType.language, path_safe=True
                    )
                    if language_path.exists():
                        _merge_tabular_language_data(workspace_dir=language_path)

                        tabular_workspace = self._job_workspace_dir / get_model_label(
                            table_lookup[table_name], ModelType.tabular, path_safe=True
                        )
                        tabular_workspace.mkdir(parents=True, exist_ok=True)
                        shutil.rmtree(tabular_workspace / "SyntheticData", ignore_errors=True)
                        shutil.move(language_path / "SyntheticData", tabular_workspace / "SyntheticData")

                finalize_method = (
                    self.execute_finalize_generation
                    if step.step_code == StepCode.finalize_generation
                    else self.execute_finalize_probing
                )
                finalize_method()

            elif step.step_code == StepCode.deliver_data:
                self.execute_deliver_data()

    def execute_finalize_generation(self):
        schema = create_generation_schema(
            generator=self._generator,
            job_workspace_dir=self._job_workspace_dir,
            step="finalize_generation",
        )

        usages = execute_step_finalize_generation(
            schema=schema,
            is_probe=False,
            job_workspace_dir=self._job_workspace_dir,
            update_progress=LocalProgressCallback(
                resource_path=self._home_dir / "synthetic-datasets" / self._synthetic_dataset.id,
                model_label=None,
                step_code=StepCode.finalize_generation,
            ),
        )
        update_total_rows(self._synthetic_dataset, usages)

    def execute_deliver_data(self):
        schema = create_generation_schema(
            generator=self._generator,
            job_workspace_dir=self._job_workspace_dir,
            step="deliver_data",
        )

        delivery = self._synthetic_dataset.delivery
        connector = (
            read_connector_from_json(self._home_dir / "connectors" / delivery.destination_connector_id)
            if delivery and delivery.destination_connector_id
            else None
        )

        execute_step_deliver_data(
            generator=self._generator,
            delivery=delivery,
            connector=connector,
            schema=schema,
            job_workspace_dir=self._job_workspace_dir,
        )

    def execute_finalize_probing(self):
        schema = create_generation_schema(
            generator=self._generator,
            job_workspace_dir=self._job_workspace_dir,
            step="finalize_generation",
        )
        # step: FINALIZE_GENERATION
        _ = execute_step_finalize_generation(
            schema=schema,
            is_probe=True,
            job_workspace_dir=self._job_workspace_dir,
        )


### JOB EXECUTION FUNCTIONS ###


def execute_training_job(generator_id: str, home_dir: Path):
    generator_dir = home_dir / "generators" / generator_id
    generator = read_generator_from_json(generator_dir)
    _set_random_state(generator.random_state)
    if generator.training_status not in [ProgressStatus.new, ProgressStatus.continue_]:
        raise ValueError("Generator has already been trained")
    _mark_in_progress(resource=generator, resource_dir=generator_dir)

    # PLAN
    plan = make_generator_execution_plan(generator)
    # EXECUTE
    execution = Execution(execution_plan=plan, generator=generator, home_dir=home_dir)
    try:
        execution.run()
        _set_overall_accuracy(generator)
        # STORE
        _move_training_artefacts(generator_dir=generator_dir, job_workspace_dir=execution._job_workspace_dir)
        # flag as DONE
        _mark_done(resource=generator, resource_dir=generator_dir)
    except Exception as e:
        _mark_failed(resource=generator, resource_dir=generator_dir)
        _LOG.error(traceback.format_exc())
        _LOG.error(f"Exception type: {type(e)}")
        _LOG.error(f"Exception message: {e}")
        raise
    finally:
        execution.clear_job_workspace()
        execution.clear_file_upload_connectors()
        write_generator_to_json(generator_dir, generator)

    try:
        _probe_random_samples(home_dir=home_dir, generator=generator)
    except Exception as e:
        _LOG.info(f"Failed to probe random samples: {e}")
        pass


def execute_generation_job(synthetic_dataset_id: str, home_dir: Path):
    synthetic_dataset_dir = home_dir / "synthetic-datasets" / synthetic_dataset_id
    synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
    _set_random_state(synthetic_dataset.random_state)
    generator_dir = home_dir / "generators" / synthetic_dataset.generator_id
    generator = read_generator_from_json(generator_dir)
    if generator.training_status != ProgressStatus.done:
        raise ValueError("Generator has not been trained yet")
    if synthetic_dataset.generation_status != ProgressStatus.new:
        raise ValueError("Synthetic Dataset has already been generated")
    _mark_in_progress(resource=synthetic_dataset, resource_dir=synthetic_dataset_dir)

    # PLAN
    plan = make_synthetic_dataset_execution_plan(generator, synthetic_dataset)
    # EXECUTE
    execution = Execution(
        execution_plan=plan,
        generator=generator,
        synthetic_dataset=synthetic_dataset,
        home_dir=home_dir,
    )
    try:
        execution.run()
        # STORE
        _move_generation_artefacts(
            synthetic_dataset_dir=synthetic_dataset_dir, job_workspace_dir=execution._job_workspace_dir
        )
        # flag as DONE
        _mark_done(resource=synthetic_dataset, resource_dir=synthetic_dataset_dir)
    except Exception:
        _mark_failed(resource=synthetic_dataset, resource_dir=synthetic_dataset_dir)
        _LOG.error(traceback.format_exc())
        raise
    finally:
        execution.clear_job_workspace()
        execution.clear_file_upload_connectors()
        write_synthetic_dataset_to_json(synthetic_dataset_dir, synthetic_dataset)


def execute_probing_job(synthetic_dataset_id: str, home_dir: Path) -> list[Probe]:
    synthetic_dataset_dir = home_dir / "synthetic-datasets" / synthetic_dataset_id
    synthetic_dataset = read_synthetic_dataset_from_json(synthetic_dataset_dir)
    _set_random_state(synthetic_dataset.random_state)
    generator_id = synthetic_dataset.generator_id
    generator_dir = home_dir / "generators" / generator_id
    generator = read_generator_from_json(generator_dir)

    # PLAN
    plan = make_synthetic_dataset_execution_plan(generator, synthetic_dataset, is_probe=True)
    # EXECUTE
    execution = Execution(
        execution_plan=plan, generator=generator, synthetic_dataset=synthetic_dataset, home_dir=home_dir
    )
    try:
        execution.run()
        # STORE
        probes = []
        for table in synthetic_dataset.tables:
            delivery_dir = execution._job_workspace_dir / "FinalizedSyntheticData" / table.name
            df = pd.read_parquet(delivery_dir / "parquet")
            probes.append(
                Probe(
                    name=table.name,
                    rows=json.loads(df.to_json(orient="records", date_format="iso", index=False)),
                )
            )
    except Exception:
        _LOG.error(traceback.format_exc())
        raise
    finally:
        execution.clear_job_workspace()
        execution.clear_file_upload_connectors()
        # no artifacts of a synthetic probe job should be left
        shutil.rmtree(synthetic_dataset_dir, ignore_errors=True)
    return probes
