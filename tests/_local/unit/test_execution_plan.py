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

from mostlyai.sdk.domain import (
    ModelEncodingType,
    Generator,
    SourceColumn,
    SourceTable,
    SourceForeignKey,
    ProgressStatus,
    GeneratorConfig,
    TaskType,
    StepCode,
)
from mostlyai.sdk._local.execution.plan import (
    ExecutionPlan,
    make_synthetic_dataset_execution_plan,
    make_generator_execution_plan,
)
from mostlyai.sdk.domain import (
    ModelConfiguration,
)


def arbitrary_model_config():
    return ModelConfiguration(
        model="some_model",
        max_sample_size=1000,
        batch_size=32,
        max_training_time=3600,
        max_epochs=100,
        max_sequence_window=512,
        enable_flexible_generation=True,
        value_protection=True,
        rare_category_replacement_method=None,
    )


def test_make_synthetic_dataset_execution_plan():
    # Create a test configuration with multiple tables and dependencies
    config = GeneratorConfig(
        **{
            "name": "test_synthetic_dataset",
            "tables": [
                {"name": "users", "primary_key": "id"},
                {
                    "name": "orders",
                    "primary_key": "id",
                    "foreign_keys": [{"column": "user_id", "referenced_table": "users", "is_context": True}],
                },
                {"name": "admins"},
                {
                    "name": "prices",
                    "foreign_keys": [{"column": "order_id", "referenced_table": "orders", "is_context": True}],
                },
                {
                    "name": "order_items",
                    "columns": [
                        {"name": "description", "model_encoding_type": "LANGUAGE_TEXT"},
                        {"name": "order_id", "model_encoding_type": "AUTO"},
                    ],
                    "foreign_keys": [{"column": "order_id", "referenced_table": "orders", "is_context": True}],
                },
            ],
        }
    )
    generator = Generator(**config.model_dump(exclude_none=True))
    execution_plan = make_synthetic_dataset_execution_plan(generator)

    expected_execution_plan = ExecutionPlan(tasks=[])
    sync_task = expected_execution_plan.add_task(TaskType.sync)
    expected_execution_plan.add_task(TaskType.generate_tabular, parent=sync_task, target_table_name="admins")
    users_task = expected_execution_plan.add_task(
        TaskType.generate_tabular, parent=sync_task, target_table_name="users"
    )
    orders_task = expected_execution_plan.add_task(
        TaskType.generate_tabular, parent=users_task, target_table_name="orders"
    )
    order_items_tabular_task = expected_execution_plan.add_task(
        TaskType.generate_tabular, parent=orders_task, target_table_name="order_items"
    )
    expected_execution_plan.add_task(
        TaskType.generate_language, parent=order_items_tabular_task, target_table_name="order_items"
    )
    expected_execution_plan.add_task(TaskType.generate_tabular, parent=orders_task, target_table_name="prices")
    post_generation_sync = expected_execution_plan.add_task(TaskType.sync)
    finalize_task = expected_execution_plan.add_task(TaskType.finalize_generation, parent=post_generation_sync)
    expected_execution_plan.add_task(TaskType.sync, parent=finalize_task)

    assert len(execution_plan.tasks) == len(expected_execution_plan.tasks)
    for actual, expected in zip(execution_plan.tasks, expected_execution_plan.tasks):
        assert actual.type == expected.type
        assert actual.target_table_name == expected.target_table_name
        actual_parent = next((t for t in execution_plan.tasks if t.id == actual.parent_task_id), None)
        expected_parent = next((t for t in expected_execution_plan.tasks if t.id == expected.parent_task_id), None)
        if actual_parent is None and expected_parent is None:
            continue
        assert actual_parent is not None
        assert expected_parent is not None
        assert actual_parent.type == expected_parent.type
        assert actual_parent.target_table_name == expected_parent.target_table_name


def test_make_generator_execution_plan():
    model_config = arbitrary_model_config()
    config = Generator(
        name="test_generator",
        training_status=ProgressStatus.in_progress,
        tables=[
            SourceTable(
                name="users",
                columns=[
                    SourceColumn(
                        name="age",
                        model_encoding_type=ModelEncodingType.tabular_numeric_digit,
                        included=True,
                    )
                ],
                tabular_model_configuration=model_config,
            ),
            SourceTable(
                name="posts",
                columns=[
                    SourceColumn(
                        name="content",
                        model_encoding_type=ModelEncodingType.language_text,
                        included=True,
                    )
                ],
                language_model_configuration=model_config,
            ),
            SourceTable(
                name="comments",
                columns=[
                    SourceColumn(
                        name="rating",
                        model_encoding_type=ModelEncodingType.tabular_numeric_digit,
                        included=True,
                    ),
                    SourceColumn(
                        name="text",
                        model_encoding_type=ModelEncodingType.language_text,
                        included=True,
                    ),
                ],
                tabular_model_configuration=model_config,
                language_model_configuration=model_config,
            ),
        ],
    )

    execution_plan = make_generator_execution_plan(config)

    expected_execution_plan = ExecutionPlan(tasks=[])
    sync_task = expected_execution_plan.add_task(TaskType.sync)
    expected_execution_plan.add_task(TaskType.train_tabular, parent=sync_task, target_table_name="users")
    expected_execution_plan.add_task(TaskType.train_language, parent=sync_task, target_table_name="posts")
    expected_execution_plan.add_task(TaskType.train_tabular, parent=sync_task, target_table_name="comments")
    expected_execution_plan.add_task(TaskType.train_language, parent=sync_task, target_table_name="comments")
    expected_execution_plan.add_task(TaskType.sync)
    # expected_execution_plan.add_task(TaskType.sync)
    # finalize_task = expected_execution_plan.add_task(TaskType.finalize_training, parent=post_training_sync)
    # expected_execution_plan.add_task(TaskType.sync, parent=finalize_task)

    assert len(execution_plan.tasks) == len(expected_execution_plan.tasks)
    for actual, expected in zip(execution_plan.tasks, expected_execution_plan.tasks):
        assert actual.type == expected.type
        assert actual.target_table_name == expected.target_table_name
        actual_parent = next((t for t in execution_plan.tasks if t.id == actual.parent_task_id), None)
        expected_parent = next((t for t in expected_execution_plan.tasks if t.id == expected.parent_task_id), None)
        if actual_parent is None and expected_parent is None:
            continue
        assert actual_parent is not None
        assert expected_parent is not None
        assert actual_parent.type == expected_parent.type
        assert actual_parent.target_table_name == expected_parent.target_table_name


def test_make_synthetic_dataset_execution_plan_with_probe():
    model_config = arbitrary_model_config()
    config = Generator(
        name="test_probe_dataset",
        training_status=ProgressStatus.in_progress,
        tables=[
            SourceTable(
                name="users",
                columns=[
                    SourceColumn(
                        name="age",
                        model_encoding_type=ModelEncodingType.tabular_numeric_digit,
                        included=True,
                    ),
                ],
                tabular_model_configuration=model_config,
            ),
            SourceTable(
                name="posts",
                columns=[
                    SourceColumn(
                        name="content",
                        model_encoding_type=ModelEncodingType.language_text,
                        included=True,
                    ),
                ],
                language_model_configuration=model_config,
                foreign_keys=[
                    SourceForeignKey(
                        column="user_id",
                        referenced_table="users",
                        is_context=True,
                    )
                ],
            ),
        ],
    )

    execution_plan = make_synthetic_dataset_execution_plan(config, is_probe=True)

    task_types = [task.type for task in execution_plan.tasks]

    assert TaskType.probe_tabular in task_types
    assert TaskType.probe_language in task_types
    assert TaskType.finalize_probing in task_types

    assert TaskType.generate_tabular not in task_types
    assert TaskType.generate_language not in task_types
    assert TaskType.finalize_generation not in task_types

    step_codes = [step for task in execution_plan.tasks for step in task.steps or []]
    assert step_codes == [StepCode.generate_data, StepCode.generate_data, StepCode.finalize_probing]
