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

from collections.abc import Callable
from functools import partial
from typing import Protocol

from rich.progress import Progress


class ProgressCallback(Protocol):
    def __call__(
        self,
        total: int | None = None,
        completed: int | None = None,
        advance: int | None = None,
        **kwargs,
    ) -> None: ...


class ProgressCallbackWrapper:
    @staticmethod
    def _wrap_progress_callback(
        update_progress: ProgressCallback | None = None, **kwargs
    ) -> tuple[ProgressCallback, Callable]:
        if not update_progress:
            rich_progress = Progress()
            rich_progress.start()
            task_id = rich_progress.add_task(**kwargs)
            update_progress = partial(rich_progress.update, task_id=task_id)
        else:
            rich_progress = None

        def teardown_progress():
            if rich_progress:
                rich_progress.refresh()
                rich_progress.stop()

        return update_progress, teardown_progress

    def update(
        self,
        total: int | None = None,
        completed: int | None = None,
        advance: int | None = None,
        **kwargs,
    ) -> None:
        self._update_progress(total=total, completed=completed, advance=advance, **kwargs)

    def __init__(self, update_progress: ProgressCallback | None = None, **kwargs):
        self._update_progress, self._teardown_progress = self._wrap_progress_callback(update_progress, **kwargs)

    def __enter__(self):
        self._update_progress(completed=0, total=1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._update_progress(completed=1, total=1)
        self._teardown_progress()
