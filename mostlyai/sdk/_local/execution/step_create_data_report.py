# Copyright 2024-2025 MOSTLY AI
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

from collections.abc import Callable
from pathlib import Path

from mostlyai.sdk._local.execution.step_create_model_report import create_report
from mostlyai.sdk.domain import Generator, ModelType, StepCode


def execute_step_create_data_report(
    *,
    generator: Generator,
    target_table_name: str,
    model_type: ModelType,
    workspace_dir: Path,
    report_credits: str = "",
    update_progress: Callable,
):
    # create model report and return metrics
    create_report(
        step_code=StepCode.create_data_report_tabular
        if model_type == ModelType.tabular
        else StepCode.create_data_report_language,
        generator=generator,
        workspace_dir=workspace_dir,
        model_type=model_type,
        target_table_name=target_table_name,
        report_credits=report_credits,
        update_progress=update_progress,
    )
