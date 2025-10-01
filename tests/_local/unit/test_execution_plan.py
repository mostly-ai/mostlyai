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

from mostlyai.sdk._local.execution.plan import (
    TRAINING_TASK_REPORT_STEPS,
    TRAINING_TASK_STEPS,
    ExecutionPlan,
    Step,
    make_generator_execution_plan,
    make_synthetic_dataset_execution_plan,
)
from mostlyai.sdk.domain import (
    Generator,
    GeneratorConfig,
    ModelConfiguration,
    ModelEncodingType,
    ProgressStatus,
    SourceColumn,
    SourceForeignKey,
    SourceTable,
    StepCode,
    SyntheticDataset,
    SyntheticTable,
    SyntheticTableConfiguration,
    TaskType,
)


def arbitrary_model_config(enable_model_report=True):
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
        enable_model_report=enable_model_report,
    )


def assert_equivalent_execution_plan(plan_a, plan_b):
    assert len(plan_a.tasks) == len(plan_b.tasks)
    for actual, expected in zip(plan_a.tasks, plan_b.tasks):
        assert actual.type == expected.type
        if not actual.steps:
            assert actual.steps == expected.steps
            continue
        assert [(step.step_code, step.target_table_name) for step in actual.steps] == [
            (step.step_code, step.target_table_name) for step in expected.steps
        ]


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
                    "tabular_model_configuration": {"enable_model_report": False},
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
                    "tabular_model_configuration": {"enable_model_report": False},
                },
            ],
        }
    )
    generator = Generator(**config.model_dump(exclude_none=True))

    synthetic_dataset = SyntheticDataset(
        id="test_synthetic_dataset",
        name="test_synthetic_dataset",
        tables=[
            SyntheticTable(
                name=table.name, configuration=SyntheticTableConfiguration(enable_data_report=table.name != "orders")
            )
            for table in generator.tables
        ],
    )

    execution_plan = make_synthetic_dataset_execution_plan(generator, synthetic_dataset)

    expected_execution_plan = ExecutionPlan(tasks=[])
    sync_task = expected_execution_plan.add_task(TaskType.sync)
    generate_task = expected_execution_plan.add_task_with_steps(
        TaskType.generate,
        parent=sync_task,
        steps=[
            Step(step_code=StepCode.generate_data_tabular, target_table_name="admins"),
            Step(step_code=StepCode.create_data_report_tabular, target_table_name="admins"),
            Step(step_code=StepCode.generate_data_tabular, target_table_name="users"),
            Step(step_code=StepCode.create_data_report_tabular, target_table_name="users"),
            Step(step_code=StepCode.generate_data_tabular, target_table_name="orders"),
            # No data report for orders table
            Step(step_code=StepCode.generate_data_tabular, target_table_name="order_items"),
            # No tabular data report for order_items table
            Step(step_code=StepCode.generate_data_language, target_table_name="order_items"),
            Step(step_code=StepCode.create_data_report_language, target_table_name="order_items"),
            Step(step_code=StepCode.generate_data_tabular, target_table_name="prices"),
            Step(step_code=StepCode.create_data_report_tabular, target_table_name="prices"),
            Step(step_code=StepCode.finalize_generation),
            Step(step_code=StepCode.deliver_data),
        ],
    )
    expected_execution_plan.add_task(TaskType.sync, parent=generate_task)

    assert_equivalent_execution_plan(execution_plan, expected_execution_plan)


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
                language_model_configuration=arbitrary_model_config(enable_model_report=False),
            ),
        ],
    )

    execution_plan = make_generator_execution_plan(config)

    expected_execution_plan = ExecutionPlan(tasks=[])
    sync_task = expected_execution_plan.add_task(TaskType.sync)

    expected_execution_plan.add_task(TaskType.train_tabular, parent=sync_task, target_table_name="users")
    expected_execution_plan.add_task(TaskType.train_language, parent=sync_task, target_table_name="posts")
    expected_execution_plan.add_task(TaskType.train_tabular, parent=sync_task, target_table_name="comments")
    expected_execution_plan.add_task(
        TaskType.train_language, parent=sync_task, target_table_name="comments", include_report=False
    )
    expected_execution_plan.add_task(TaskType.sync)

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
        # verify report steps are included/excluded based on configuration
        if actual.steps:
            actual_step_codes = [s.step_code for s in actual.steps]
            expected_step_codes = [s.step_code for s in expected.steps]
            assert actual_step_codes == expected_step_codes
            if actual.target_table_name == "comments" and actual.type == TaskType.train_language:
                assert actual_step_codes == TRAINING_TASK_STEPS
            else:
                assert actual_step_codes == TRAINING_TASK_STEPS + TRAINING_TASK_REPORT_STEPS


def test_make_synthetic_dataset_execution_plan_with_probe():
    model_config = arbitrary_model_config()
    generator = Generator(
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
    synthetic_dataset = SyntheticDataset(
        tables=[
            SyntheticTable(name=table.name, configuration=SyntheticTableConfiguration()) for table in generator.tables
        ],
    )

    execution_plan = make_synthetic_dataset_execution_plan(generator, synthetic_dataset, is_probe=True)

    expected_execution_plan = ExecutionPlan(tasks=[])
    sync_task = expected_execution_plan.add_task(TaskType.sync)
    generate_task = expected_execution_plan.add_task_with_steps(
        TaskType.probe,
        parent=sync_task,
        steps=[
            Step(step_code=StepCode.generate_data_tabular, target_table_name="users"),
            Step(step_code=StepCode.generate_data_language, target_table_name="posts"),
            Step(step_code=StepCode.finalize_probing),
        ],
    )
    expected_execution_plan.add_task(TaskType.sync, parent=generate_task)

    assert_equivalent_execution_plan(execution_plan, expected_execution_plan)
