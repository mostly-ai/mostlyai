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
import uuid
from collections import deque

from pydantic import BaseModel, ConfigDict, Field

from mostlyai.sdk.domain import (
    Generator,
    ModelEncodingType,
    ModelType,
    SourceTable,
    StepCode,
    SyntheticDataset,
    TaskType,
)

TABULAR_MODEL_ENCODING_TYPES = [v for v in ModelEncodingType if v.startswith(ModelType.tabular)] + [
    ModelEncodingType.auto
]
LANGUAGE_MODEL_ENCODING_TYPES = [v for v in ModelEncodingType if v.startswith(ModelType.language)]

TRAINING_TASK_STEPS: list[StepCode] = [
    StepCode.pull_training_data,
    StepCode.analyze_training_data,
    StepCode.encode_training_data,
    StepCode.train_model,
]
TRAINING_TASK_REPORT_STEPS: list[StepCode] = [
    StepCode.generate_model_report_data,
    StepCode.create_model_report,
]
FINALIZE_TRAINING_TASK_STEPS: list[StepCode] = [
    StepCode.finalize_training,
]
FINALIZE_GENERATION_TASK_STEPS: list[StepCode] = [
    StepCode.finalize_generation,
    StepCode.deliver_data,
]


def has_tabular_model(table: SourceTable) -> bool:
    return table.tabular_model_configuration is not None


def has_language_model(table: SourceTable) -> bool:
    return table.language_model_configuration is not None


def get_model_type_generation_steps_map(
    enable_data_report: bool, table: SourceTable
) -> dict[ModelType, list[StepCode]]:
    steps = {
        ModelType.tabular: [StepCode.generate_data_tabular],
        ModelType.language: [StepCode.generate_data_language],
    }
    if enable_data_report and has_tabular_model(table):
        enable_tabular_model_report = table.tabular_model_configuration.enable_model_report
        if enable_tabular_model_report:
            steps[ModelType.tabular].append(StepCode.create_data_report_tabular)
    if enable_data_report and has_language_model(table):
        enable_language_model_report = table.language_model_configuration.enable_model_report
        if enable_language_model_report:
            steps[ModelType.language].append(StepCode.create_data_report_language)
    return steps


class Step(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_code: StepCode = Field(alias="stepCode")
    target_table_name: str | None = Field(None, alias="targetTableName", title="Target Table Name")


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    parent_task_id: str | None = Field(None, alias="parentTaskId", title="Parent Task Id")
    target_table_name: str | None = Field(None, alias="targetTableName", title="Target Table Name")
    type: TaskType
    steps: list[Step] | None


class ExecutionPlan(BaseModel):
    tasks: list[Task]

    def add_task(
        self,
        task_type: TaskType,
        parent: Task | None = None,
        target_table_name: str | None = None,
        include_report: bool = True,
    ) -> Task:
        def get_steps(task_type: TaskType) -> list[Step] | None:
            match task_type:
                case TaskType.train_tabular | TaskType.train_language:
                    return [
                        Step(step_code=code)
                        for code in TRAINING_TASK_STEPS + (TRAINING_TASK_REPORT_STEPS if include_report else [])
                    ]
                case TaskType.finalize_training:
                    return [Step(step_code=code) for code in FINALIZE_TRAINING_TASK_STEPS]
                case _:
                    return None

        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            parent_task_id=parent.id if parent else None,
            target_table_name=target_table_name,
            steps=get_steps(task_type),
        )
        self.tasks.append(task)
        return task

    def add_task_with_steps(self, task_type: TaskType, parent: Task | None = None, steps: list[Step] = None) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            parent_task_id=parent.id if parent else None,
            steps=steps or [],
        )
        self.tasks.append(task)
        return task


def make_generator_execution_plan(generator: Generator) -> ExecutionPlan:
    execution_plan = ExecutionPlan(tasks=[])
    sync_task = execution_plan.add_task(TaskType.sync)
    for table in generator.tables:
        if has_tabular_model(table):
            execution_plan.add_task(
                TaskType.train_tabular,
                parent=sync_task,
                target_table_name=table.name,
                include_report=table.tabular_model_configuration.enable_model_report,
            )
        if has_language_model(table):
            execution_plan.add_task(
                TaskType.train_language,
                parent=sync_task,
                target_table_name=table.name,
                include_report=table.language_model_configuration.enable_model_report,
            )
    execution_plan.add_task(TaskType.sync)
    # post_training_sync = execution_plan.add_task(TaskType.sync)
    # finalize_task = execution_plan.add_task(TaskType.finalize_training, parent=post_training_sync)
    # execution_plan.add_task(TaskType.sync, parent=finalize_task)
    return execution_plan


def make_synthetic_dataset_execution_plan(
    generator: Generator, synthetic_dataset: SyntheticDataset, is_probe: bool = False
) -> ExecutionPlan:
    execution_plan = ExecutionPlan(tasks=[])
    sync_task = execution_plan.add_task(TaskType.sync)
    generate_task_type = TaskType.probe if is_probe else TaskType.generate
    finalize_step_code = StepCode.finalize_probing if is_probe else StepCode.finalize_generation
    generate_steps = []

    def add_generation_steps(table: SourceTable):
        synthetic_table = next(t for t in synthetic_dataset.tables if t.name == table.name)
        enable_data_report = synthetic_table.configuration.enable_data_report
        if has_tabular_model(table):
            enable_tabular_model_report = table.tabular_model_configuration.enable_model_report
            steps = [Step(step_code=StepCode.generate_data_tabular, target_table_name=table.name)]
            if not is_probe and enable_data_report and enable_tabular_model_report:
                steps.append(Step(step_code=StepCode.create_data_report_tabular, target_table_name=table.name))
            generate_steps.extend(steps)

        if has_language_model(table):
            enable_language_model_report = table.language_model_configuration.enable_model_report
            steps = [Step(step_code=StepCode.generate_data_language, target_table_name=table.name)]
            if not is_probe and enable_data_report and enable_language_model_report:
                steps.append(Step(step_code=StepCode.create_data_report_language, target_table_name=table.name))
            generate_steps.extend(steps)

    # Identify root tables (tables without a foreign key referencing them)
    root_tables = sorted(
        (table for table in generator.tables if not any(fk.is_context for fk in table.foreign_keys or [])),
        key=lambda t: t.name,
    )

    # Process each root table and its subtree
    for root_table in root_tables:
        add_generation_steps(root_table)

        # Traverse child tables using BFS
        queue = deque([root_table.name])
        while queue:
            current_table_name = queue.popleft()

            child_tables = sorted(
                (
                    table
                    for table in generator.tables
                    if any(
                        fk.referenced_table == current_table_name and fk.is_context for fk in table.foreign_keys or []
                    )
                ),
                key=lambda t: t.name,
            )

            for child_table in child_tables:
                add_generation_steps(child_table)
                queue.append(child_table.name)

    # Add last common step(s)
    if generate_steps:
        generate_steps.append(Step(step_code=finalize_step_code))
        if not is_probe:
            generate_steps.append(Step(step_code=StepCode.deliver_data))
        generate_task = execution_plan.add_task_with_steps(generate_task_type, steps=generate_steps, parent=sync_task)
        execution_plan.add_task(TaskType.sync, parent=generate_task)

    return execution_plan
