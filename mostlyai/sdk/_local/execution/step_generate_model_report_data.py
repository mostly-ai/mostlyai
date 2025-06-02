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

import pandas as pd

from mostlyai.sdk.domain import ModelType

_LOG = logging.getLogger(__name__)


def execute_step_generate_model_report_data(
    *,
    workspace_dir: Path,
    model_type: ModelType,
    update_progress: Callable,
):
    # import ENGINE here to avoid pre-mature loading of large ENGINE dependencies
    from mostlyai import engine
    from mostlyai.engine._workspace import Workspace

    # determine max sample size for generated report samples
    workspace = Workspace(workspace_dir)
    tgt_stats = workspace.tgt_stats.read()
    max_sample_size = qa_sample_size_heuristic(tgt_stats=tgt_stats, model_type=model_type)

    # pull context data for report generation (if applicable)
    has_context = workspace.ctx_stats.path.exists()
    if has_context:
        ctx_stats = workspace.ctx_stats.read()
        ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key")
        ctx_input_path = workspace_dir / "report-ctx-data"
        _pull_context_for_report_generation(
            ctx_data_path=workspace.ctx_data_path,
            output_path=ctx_input_path,
            max_sample_size=max_sample_size,
            ctx_primary_key=ctx_primary_key,
        )
        ctx_data = pd.read_parquet(ctx_input_path)
    else:
        ctx_data = None

    # call GENERATE
    engine.generate(
        ctx_data=ctx_data,
        sample_size=max_sample_size,
        workspace_dir=workspace_dir,
        update_progress=update_progress,
    )


def qa_sample_size_heuristic(tgt_stats: dict, model_type: ModelType) -> int:
    # import ENGINE here to avoid pre-mature loading of large ENGINE dependencies
    from mostlyai.engine._common import get_cardinalities, get_sequence_length_stats

    if model_type == ModelType.language:
        return 1_000
    trn_sample_size = tgt_stats["no_of_training_records"] + tgt_stats["no_of_validation_records"]
    no_tgt_sub_columns = len(get_cardinalities(tgt_stats))
    tgt_q50_seqlen = get_sequence_length_stats(tgt_stats)["median"]
    data_points = no_tgt_sub_columns * tgt_q50_seqlen
    if data_points > 1_000:
        gen_sample_size = 10_000
    else:
        gen_sample_size = 100_000
    return min(gen_sample_size, trn_sample_size)


def _pull_context_for_report_generation(
    *, ctx_data_path: Path, output_path: Path, max_sample_size: int, ctx_primary_key: str
):
    ctx_trn_files = sorted(ctx_data_path.glob("part.*-trn.parquet"))
    ctx_val_files = sorted(ctx_data_path.glob("part.*-val.parquet"))
    # fetch keys alone first
    ctx_trn_keys = pd.concat([pd.read_parquet(f, columns=[ctx_primary_key]) for f in ctx_trn_files], ignore_index=True)
    ctx_val_keys = pd.concat([pd.read_parquet(f, columns=[ctx_primary_key]) for f in ctx_val_files], ignore_index=True)
    ctx_trn_keys = ctx_trn_keys.sample(frac=1).reset_index(drop=True)
    ctx_val_keys = ctx_val_keys.sample(frac=1).reset_index(drop=True)
    # attempt to balance the training and validation sets
    trn_keys = int(max_sample_size * 0.50)
    val_keys = max_sample_size - trn_keys
    keys = pd.concat(
        [
            ctx_trn_keys[:trn_keys].assign(set="trn"),
            ctx_val_keys[:val_keys].assign(set="val"),
            ctx_trn_keys[trn_keys:].assign(set="trn"),
            ctx_val_keys[val_keys:].assign(set="val"),
        ],
        ignore_index=True,
    )
    keys = keys.head(max_sample_size)
    ctx_trn_keys = keys[keys["set"] == "trn"][[ctx_primary_key]]
    ctx_val_keys = keys[keys["set"] == "val"][[ctx_primary_key]]
    # fetch rest of the context data
    df_trn_ctx = pd.concat([pd.read_parquet(f).merge(ctx_trn_keys, on=ctx_primary_key) for f in ctx_trn_files])
    df_val_ctx = pd.concat([pd.read_parquet(f).merge(ctx_val_keys, on=ctx_primary_key) for f in ctx_val_files])
    df_trn_ctx = df_trn_ctx.reset_index(drop=True)
    df_val_ctx = df_val_ctx.reset_index(drop=True)
    output_path.mkdir(parents=True, exist_ok=True)
    df_trn_ctx.to_parquet(output_path / f"part.{0:06}.{0:06}-trn.parquet")
    df_val_ctx.to_parquet(output_path / f"part.{0:06}.{0:06}-val.parquet")
    _LOG.info(f"pulled context data for model report ({len(df_trn_ctx)=:,} {len(df_val_ctx)=:,})")
