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

import logging
import time
from pathlib import Path
from typing import Any

from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.progress_callback import ProgressCallbackWrapper
from mostlyai.sdk._data.pull_utils import handle_workspace_dir, pull_split
from mostlyai.sdk.domain import ModelType

_LOG = logging.getLogger(__name__)


def pull_context(
    *,
    tgt: str,
    schema: Schema,
    max_sample_size: int | None = None,
    model_type: str | ModelType = ModelType.tabular,
    workspace_dir: str | Path = "engine-ws",
):
    t0 = time.time()
    workspace_dir = Path(workspace_dir)
    model_type = ModelType(model_type)
    if tgt not in schema.tables:
        raise ValueError(f"table '{tgt}' not defined in schema")
    schema.preprocess_schema_before_pull()
    # gather context_tables
    context_tables = schema.get_context_tables(tgt)
    _LOG.info(f"context_tables (size: {len(context_tables)}): {context_tables}")
    # handle workspace_dir
    workspace_dir = handle_workspace_dir(workspace_dir=workspace_dir)
    # ensure that max_sample_size is a positive integer, if given
    if max_sample_size is not None:
        max_sample_size = max(1, max_sample_size)
    # log arguments
    _LOG.info(f"tgt: {tgt}")
    _LOG.info(f"model_type: {model_type}")
    _LOG.info(f"max_sample_size: {max_sample_size}")

    def update_progress(*args: Any, **kwargs: Any) -> None: ...

    pull_split(
        tgt=tgt,
        schema=schema,
        trn_val_split=None,
        model_type=model_type,
        do_ctx_only=True,
        workspace_dir=workspace_dir,
        progress=ProgressCallbackWrapper(update_progress=update_progress),
    )

    _LOG.info(f"pull_context total time: {time.time() - t0:.2f}s")
