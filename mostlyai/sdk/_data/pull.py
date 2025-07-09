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
import shutil
import time
from pathlib import Path

from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.progress_callback import ProgressCallback, ProgressCallbackWrapper
from mostlyai.sdk._data.pull_utils import (
    handle_workspace_dir,
    pull_fetch,
    pull_keys,
    pull_split,
    remake_schema_after_pull_fetch,
)
from mostlyai.sdk.domain import ModelType

_LOG = logging.getLogger(__name__)


def pull(
    *,
    tgt: str,
    schema: Schema,
    model_type: str | ModelType = ModelType.tabular,
    max_sample_size: int | None = None,
    trn_val_split: float | None = 0.8,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
):
    t0 = time.time()
    with ProgressCallbackWrapper(update_progress, description="Pull training data") as progress:
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

        # initialize progress counter
        tbl_count_rows = 0
        tbl_count_rows += schema.tables[tgt].row_count
        for ctx_table in schema.get_context_tables(tgt):
            tbl_count_rows += schema.tables[ctx_table].row_count
        progress_plan = 1000
        progress_fetch = tbl_count_rows
        progress_split = tbl_count_rows
        progress.update(completed=0, total=progress_plan + progress_fetch + progress_split + 1)

        keys = pull_keys(
            tgt=tgt,
            schema=schema,
            max_sample_size=max_sample_size,
            model_type=model_type,
        )
        progress.update(advance=progress_plan)

        pull_fetch(
            tgt=tgt,
            schema=schema,
            keys=keys,
            max_sample_size=max_sample_size,
            workspace_dir=workspace_dir,
            progress=progress,
        )
        schema = remake_schema_after_pull_fetch(tgt=tgt, schema=schema, workspace_dir=workspace_dir)

        pull_split(
            tgt=tgt,
            schema=schema,
            trn_val_split=trn_val_split,
            model_type=model_type,
            do_ctx_only=False,
            workspace_dir=workspace_dir,
            progress=progress,
        )

        _LOG.info("clean up temporary fetch directory")
        shutil.rmtree(workspace_dir / "__PULL_FETCH", ignore_errors=True)
    _LOG.info(f"pull total time: {time.time() - t0:.2f}s")
