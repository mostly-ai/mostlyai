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

import datetime
import math
from pathlib import Path

from pydantic import BaseModel

from mostlyai.sdk._local.storage import read_job_progress_from_json, write_job_progress_to_json
from mostlyai.sdk.domain import ProgressStatus, StepCode


def get_current_utc_time() -> datetime.datetime:
    # always use UTC time for platform compatibility
    return datetime.datetime.now(datetime.timezone.utc)


class LocalProgressCallback:
    def __init__(self, resource_path: Path, model_label: str | None, step_code: StepCode, **kwargs):
        self.resource_path = resource_path
        self.model_label = model_label
        self.step_code = step_code
        # total count of progress units
        self._total_count = 1
        # buffer for increase_by
        self._increase_by = 0
        # start time of execution
        self._start_execution_time = None
        # time of last progress update
        self._last_send_progress_time = None
        self._total = None

        self.job_progress = read_job_progress_from_json(self.resource_path)

    def _check_elapsed_interval(self):
        now = get_current_utc_time()
        if self._start_execution_time is None:
            self._start_execution_time = now
            return True
        if self._last_send_progress_time is None:
            self._last_send_progress_time = now
            return True

        time_since_start = now - self._start_execution_time
        time_since_last_progress = now - self._last_send_progress_time
        # during first 100secs, send progress every second; after that, every 5secs
        if time_since_start.total_seconds() < 100:
            return time_since_last_progress.total_seconds() > 1
        else:
            return time_since_last_progress.total_seconds() > 5

    def _set_total_count(self, total_count: int | None) -> int:
        # ensure that there is at least 1 progress unit
        if total_count is not None:
            self._total_count = max(1, math.ceil(total_count))
        return self._total_count

    def _update_increase_by(self, increase_by: int | None) -> int:
        if increase_by is not None:
            self._increase_by += increase_by
        return self._increase_by

    def __call__(
        self,
        total: int | None = None,
        completed: int | None = None,
        advance: int | None = None,
        message: dict | BaseModel | None = None,
        **kwargs,
    ):
        if isinstance(message, BaseModel):
            message = message.model_dump(mode="json")
        now = get_current_utc_time()
        if completed == 0:
            self._start_execution_time = now
        if total is not None:
            self._total = total

        elapsed_enough_time = self._check_elapsed_interval()

        progress_step = next(
            s for s in self.job_progress.steps if s.step_code == self.step_code and s.model_label == self.model_label
        )
        if progress_step.start_date is None:
            progress_step.start_date = now
            progress_step.status = ProgressStatus.in_progress
        if advance is not None:
            progress_step.progress.value += advance
        if completed is not None:
            progress_step.progress.value = completed
        if total is not None:
            progress_step.progress.max = total
        if progress_step.progress.value >= progress_step.progress.max:
            progress_step.end_date = now
            progress_step.status = ProgressStatus.done
        if message is not None:
            if progress_step.messages is None:
                progress_step.messages = []
            progress_step.messages.append(message)
        self.job_progress.steps = [
            progress_step if (s.model_label == self.model_label and s.step_code == self.step_code) else s
            for s in self.job_progress.steps
        ]
        self.job_progress.progress.value = len([s for s in self.job_progress.steps if s.status == ProgressStatus.done])
        self.job_progress.progress.max = len(self.job_progress.steps)  # of steps
        if self.job_progress.start_date is None:
            self.job_progress.start_date = now
        # NOTE: do not set job status to DONE here, that should happen as the very last thing
        # in order for `job_wait` not to finish polling prematurely

        # send progress if we are DONE, or if we have a message to pass,
        # or if enough time has passed since last progress update
        total_count = self._set_total_count(total)
        increase_by = self._update_increase_by(advance)
        if (
            message is not None
            or (completed is not None and completed >= total_count)
            or (completed is not None and elapsed_enough_time)
            or (increase_by > 0 and elapsed_enough_time)
        ):
            write_job_progress_to_json(self.resource_path, self.job_progress)

        return {}
