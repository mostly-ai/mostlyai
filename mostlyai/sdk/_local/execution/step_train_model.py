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
import logging
from collections.abc import Callable
from pathlib import Path

from mostlyai.sdk.domain import Generator, ModelType

_LOG = logging.getLogger(__name__)


def execute_step_train_model(
    *,
    generator: Generator,
    model_type: ModelType,
    target_table_name: str,
    restarts: int,
    workspace_dir: Path,
    update_progress: Callable,
    upload_model_data_callback: Callable | None,
):
    # import ENGINE here to avoid pre-mature loading of large ENGINE dependencies
    from mostlyai import engine
    from mostlyai.engine.domain import DifferentialPrivacyConfig, ModelStateStrategy

    _LOG.info(f"mostlyai-engine: {engine.__version__}")

    # fetch model_config
    tgt_table = next(t for t in generator.tables if t.name == target_table_name)
    if model_type == ModelType.language:
        model_config = tgt_table.language_model_configuration
    else:
        model_config = tgt_table.tabular_model_configuration

    # convert from SDK domain to ENGINE domain
    if model_config.differential_privacy:
        differential_privacy = DifferentialPrivacyConfig(**model_config.differential_privacy.model_dump())
    else:
        differential_privacy = None

    # ensure disallowed arguments are set to None
    if model_type == ModelType.language:
        max_sequence_window = None
    else:  # model_type == ModelType.tabular
        max_sequence_window = model_config.max_sequence_window

    # call TRAIN
    engine.train(
        model=model_config.model,
        max_training_time=model_config.max_training_time,
        max_epochs=model_config.max_epochs,
        batch_size=model_config.batch_size,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        enable_flexible_generation=model_config.enable_flexible_generation,
        max_sequence_window=max_sequence_window,
        differential_privacy=differential_privacy,
        model_state_strategy=ModelStateStrategy.resume if restarts > 0 else ModelStateStrategy.reuse,
        workspace_dir=workspace_dir,
        upload_model_data_callback=upload_model_data_callback,
        update_progress=update_progress,
    )
