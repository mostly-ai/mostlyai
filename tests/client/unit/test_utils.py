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

import base64
import io
import math
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch, Mock, ANY

import pandas as pd
import pytest
from rich.console import Console

from mostlyai.sdk._local.progress import get_current_utc_time
from mostlyai.sdk.domain import (
    JobProgress,
    ProgressStatus,
    ProgressStep,
    ProgressValue,
    StepCode,
    Generator,
    Metadata,
    SourceTable,
    SyntheticDatasetConfig,
    SyntheticTableConfig,
    SyntheticProbeConfig,
)
from mostlyai.sdk.client._base_utils import (
    convert_to_base64,
    read_table_from_path,
)
from mostlyai.sdk.client._utils import (
    job_wait,
    harmonize_sd_config,
)

UTILS_MODULE = "mostlyai.sdk.utils"


def test_convert_to_base64():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    base64_str = convert_to_base64(df)

    assert isinstance(base64_str, str)

    # Decode the Base64 string back to a DataFrame
    decoded_bytes = base64.b64decode(base64_str)
    decoded_buffer = io.BytesIO(decoded_bytes)
    decoded_df = pd.read_parquet(decoded_buffer)

    # Compare the original DataFrame with the decoded one
    pd.testing.assert_frame_equal(df, decoded_df)


def test_read_table_from_path():
    # Create a temporary CSV file for testing
    delimiters = ",;|\t' :"
    for d in delimiters:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            df.to_csv(tmp.name, index=False, sep=d)
            tmp_path = tmp.name

        name, read_df = read_table_from_path(tmp_path)

        assert name == Path(tmp_path).stem
        pd.testing.assert_frame_equal(read_df, df)


@pytest.mark.skip("Fails on remote during CI")
def test__job_wait():
    # Timeline in seconds with job and step progression:
    # | Time (seconds):   | 0 | 1 | 2 | 3 | 4 | 5 |
    # |-------------------|---|---|---|---|---|---|
    # | Job Progress:     | 0 | 1 | 2 | 3 | 4 | 5 | (max = 5)
    # | Step 1 Progress:  | 0 | 1 | 2 | 3 |   |   | (max = 3, starts at 0s)
    # | Step 2 Progress:  |   | 0 | 1 | 2 | 3 |   | (max = 3, starts at 1s)
    # | Step 3 Progress:  |   |   |   | 0 | 1 | 2 | (max = 2, starts at 3s)

    # Create initial job state
    job_start_date = get_current_utc_time()
    job = JobProgress(
        id="job1",
        start_date=job_start_date,
        progress=ProgressValue(max=5, value=0),
        status=ProgressStatus.queued,
        steps=[
            ProgressStep(
                id="step1",
                status=ProgressStatus.queued,
                step_code=StepCode.train_model,
                progress=ProgressValue(value=0, max=3),
            ),
            ProgressStep(
                id="step2",
                status=ProgressStatus.queued,
                step_code=StepCode.train_model,
                progress=ProgressValue(value=0, max=3),
            ),
            ProgressStep(
                id="step3",
                status=ProgressStatus.queued,
                step_code=StepCode.train_model,
                progress=ProgressValue(value=0, max=2),
            ),
        ],
    )

    # Callback function to update the job state

    def get_job_progress():
        current_time = get_current_utc_time()
        elapsed = (current_time - job_start_date).total_seconds()

        # Set start times for each step
        step_start_times = [0, 1, 3]

        for i, step in enumerate(job.steps):
            step_elapsed = elapsed - step_start_times[i]
            if step_elapsed >= 0:
                if step.status == ProgressStatus.queued:
                    # print(f"Starting {step.id}")
                    step.status = ProgressStatus.in_progress
                    step.start_date = current_time

                # Update progress and status
                if step.status == ProgressStatus.in_progress:
                    step.progress.value = min(math.floor(step_elapsed), step.progress.max)
                    # print(f"{step.id} progress {step.progress.value}")
                    if step.progress.value >= step.progress.max:
                        # print(f"Stopping {step.id}")
                        step.status = ProgressStatus.done
                        step.end_date = current_time

        # Update overall job progress
        job.progress.value = min(math.floor(elapsed), job.progress.max)
        if job.progress.value >= job.progress.max:
            job.status = ProgressStatus.done
            job.end_date = current_time

        return job

    console = Console()
    with console.capture() as capture, patch(f"{UTILS_MODULE}.rich._console", console):
        job_wait(get_job_progress, interval=1)

    actual_lines = [line.strip() for line in capture.get().splitlines()]
    expected_lines = [
        "Overall job progress       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05",
        "Step common TRAIN_MODEL 💎 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03",
        "Step common TRAIN_MODEL 💎 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03",
        "Step common TRAIN_MODEL 💎 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02",
    ]
    assert actual_lines == expected_lines


@pytest.fixture
def simple_sd_config():
    return SyntheticDatasetConfig(tables=[SyntheticTableConfig(name="subject")])


def test__harmonize_sd_config(simple_sd_config):
    mock_get_generator = Mock(
        side_effect=lambda id: Generator(
            id=id,
            training_status=ProgressStatus.done,
            metadata=Metadata(),
            tables=[SourceTable(id=str(uuid.uuid4()), name="table", columns=[])],
        )
    )

    # case 1: existing config
    id = str(uuid.uuid4())
    config = harmonize_sd_config(
        generator=id,
        get_generator=mock_get_generator,
        config=simple_sd_config,
        config_type=SyntheticDatasetConfig,
    )
    assert config == SyntheticDatasetConfig(generator_id=id, tables=[SyntheticTableConfig(name="subject")])

    # case 2: existing config
    id2 = str(uuid.uuid4())
    config = harmonize_sd_config(
        generator=id2,
        get_generator=mock_get_generator,
        size=1234,
        seed=pd.DataFrame(),
        config_type=SyntheticProbeConfig,
    )
    assert isinstance(config, SyntheticProbeConfig)
    assert config.generator_id == id2
    assert len(config.tables) == 1
    table = config.tables[0]
    assert table.name == "table"
    assert table.configuration.sample_size == 1234
    assert table.configuration.sample_seed_data == ANY
    assert table.configuration.sample_seed_dict is None
