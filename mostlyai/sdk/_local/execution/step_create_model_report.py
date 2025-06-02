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
import random
from collections.abc import Callable
from itertools import accumulate, takewhile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as papqt

from mostlyai import qa
from mostlyai.engine._workspace import Workspace
from mostlyai.sdk._data.util.common import TABLE_COLUMN_INFIX, TEMPORARY_PRIMARY_KEY, strip_column_prefix
from mostlyai.sdk._local.execution.step_generate_model_report_data import qa_sample_size_heuristic
from mostlyai.sdk._local.storage import get_model_label
from mostlyai.sdk.domain import Generator, ModelMetrics, ModelType, StepCode

_LOG = logging.getLogger(__name__)


def execute_step_create_model_report(
    *,
    generator: Generator,
    model_type: ModelType,
    target_table_name: str,
    workspace_dir: Path,
    report_credits: str = "",
    update_progress: Callable,
) -> ModelMetrics | None:
    # create model report and return metrics
    metrics = create_report(
        step_code=StepCode.create_model_report,
        generator=generator,
        workspace_dir=workspace_dir,
        model_type=model_type,
        target_table_name=target_table_name,
        report_credits=report_credits,
        update_progress=update_progress,
    )
    return metrics


def create_report(
    *,
    step_code: StepCode,
    generator: Generator,
    workspace_dir: Path,
    model_type: ModelType,
    target_table_name: str,
    report_credits: str = "",
    update_progress: Callable,
) -> ModelMetrics | None:
    _LOG.info(f"mostlyai-qa: {qa.__version__}")

    workspace = Workspace(workspace_dir)
    tgt_table = next(t for t in generator.tables if t.name == target_table_name)

    # only consider non-key tgt columns for QA
    tgt_fks = [fk.column for fk in (tgt_table.foreign_keys or [])]
    tgt_columns = [c.name for c in tgt_table.columns if c not in tgt_fks and c.name != tgt_table.primary_key]
    # calculate sample size
    tgt_stats = workspace.tgt_stats.read()
    sample_size = qa_sample_size_heuristic(tgt_stats=tgt_stats, model_type=model_type)
    # handle context
    has_context = workspace.ctx_stats.path.exists()
    if model_type == ModelType.tabular:
        if has_context:  # TABULAR model for sequential table
            ctx_table = next(
                (
                    t
                    for t in generator.tables
                    if t.name
                    == next((fk.referenced_table for fk in (tgt_table.foreign_keys or []) if fk.is_context), None)
                ),
                None,
            )
            ctx_table_name = ctx_table.name
            ctx_primary_key = f"{ctx_table.name}{TABLE_COLUMN_INFIX}{ctx_table.primary_key}"
            tgt_context_key = next(fk for fk in tgt_table.foreign_keys if fk.is_context).column
            # only consider non-key parent ctx columns for QA
            ctx_fks = [fk.column for fk in (ctx_table.foreign_keys or [])]
            ctx_columns = [
                f"{ctx_table.name}{TABLE_COLUMN_INFIX}{c.name}"
                for c in ctx_table.columns
                if c.name not in ctx_fks and c.name != ctx_primary_key
            ]
        else:  # TABULAR model for flat table
            ctx_table_name = None
            ctx_primary_key = None
            tgt_context_key = None
            ctx_columns = None
    else:  # model_type == ModelType.language
        ctx_table = next(
            (
                t
                for t in generator.tables
                if t.name == next((fk.referenced_table for fk in (tgt_table.foreign_keys or []) if fk.is_context), None)
            ),
            None,
        )
        ctx_table_name = ctx_table.name if ctx_table else tgt_table.name
        ctx_primary_key = f"{tgt_table.name}{TABLE_COLUMN_INFIX}{TEMPORARY_PRIMARY_KEY}"
        tgt_context_key = TEMPORARY_PRIMARY_KEY
        ctx_columns = []
        if ctx_table:  # LANGUAGE model for sequential table
            # only consider non-key parent ctx columns for QA
            ctx_fks = [fk.column for fk in (ctx_table.foreign_keys or [])]
            ctx_columns += [
                f"{ctx_table.name}{TABLE_COLUMN_INFIX}{c.name}"
                for c in ctx_table.columns
                if c.name not in ctx_fks and c.name != ctx_primary_key
            ]
        # consider TABULAR tgt columns a context
        ctx_columns += [f"{target_table_name}{TABLE_COLUMN_INFIX}{c}" for c in tgt_columns]

    pull_kwargs = dict(
        ctx_primary_key=ctx_primary_key if has_context else None,
        tgt_context_key=tgt_context_key if has_context else None,
        ctx_columns=ctx_columns if has_context else None,
        tgt_columns=tgt_columns,
        ctx_table_name=ctx_table_name if has_context else None,
        max_sample_size=sample_size,
        max_sequence_length=1_000,
    )
    if step_code == StepCode.create_model_report:
        report_ctx_input_path = workspace_dir / "report-ctx-data"
        syn_tgt_data, syn_ctx_data = pull_data_for_report(
            tgt_data=workspace.generated_data.fetch_all(),
            ctx_data=sorted(report_ctx_input_path.glob("part.*.parquet")) if has_context else None,
            **pull_kwargs,
        )
        trn_tgt_data, trn_ctx_data = pull_data_for_report(
            tgt_data=workspace.tgt_trn_data.fetch_all(),
            ctx_data=sorted(report_ctx_input_path.glob("part.*-trn.parquet")) if has_context else None,
            **pull_kwargs,
        )
        hol_tgt_data, hol_ctx_data = pull_data_for_report(
            tgt_data=workspace.tgt_val_data.fetch_all(),
            ctx_data=sorted(report_ctx_input_path.glob("part.*-val.parquet")) if has_context else None,
            **pull_kwargs,
        )
    else:  # step_code == StepCode.create_data_report_*:
        syn_tgt_data, syn_ctx_data = pull_data_for_report(
            tgt_data=workspace.generated_data.fetch_all(),
            ctx_data=workspace.ctx_data.fetch_all() if has_context else None,
            **pull_kwargs,
        )
        trn_tgt_data, trn_ctx_data = None, None
        hol_tgt_data, hol_ctx_data = None, None

    if ctx_primary_key and ctx_table_name:
        ctx_primary_key = strip_column_prefix(prefixed_data=ctx_primary_key, table_name=ctx_table_name)

    if step_code == StepCode.create_model_report:
        # generate Model QA report
        (workspace_dir / "ModelQAReports").mkdir(exist_ok=True, parents=True)
        _, metrics = qa.report(
            syn_tgt_data=syn_tgt_data,
            trn_tgt_data=trn_tgt_data,
            hol_tgt_data=hol_tgt_data,
            syn_ctx_data=syn_ctx_data,
            trn_ctx_data=trn_ctx_data,
            hol_ctx_data=hol_ctx_data,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            report_path=workspace_dir
            / "ModelQAReports"
            / f"{get_model_label(tgt_table, model_type, path_safe=True)}.html",
            report_title="Model Report",
            report_subtitle=f" for {target_table_name[:30]}:{model_type.name}",
            report_credits=report_credits,
            max_sample_size_accuracy=100_000,
            max_sample_size_embeddings=10_000,
            statistics_path=workspace_dir / "ModelQAStatistics",
            update_progress=update_progress,
        )
        # convert metrics from QA domain to SDK domain (if applicable)
        metrics = ModelMetrics(**metrics.model_dump()) if metrics else None
    else:  # if step_code == StepCode.create_data_report_*:
        # generate Data QA report
        (workspace_dir / "DataQAReports").mkdir(exist_ok=True, parents=True)
        _ = qa.report_from_statistics(
            syn_tgt_data=syn_tgt_data,
            syn_ctx_data=syn_ctx_data,
            statistics_path=workspace_dir / "ModelQAStatistics",
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=tgt_context_key,
            report_path=workspace_dir
            / "DataQAReports"
            / f"{get_model_label(tgt_table, model_type, path_safe=True)}.html",
            report_title="Data Report",
            report_subtitle=f" for {target_table_name[:30]}:{model_type.name}",
            report_credits=report_credits,
            max_sample_size_accuracy=100_000,
            update_progress=update_progress,
        )
        metrics = None
    return metrics


def pull_data_for_report(
    *,
    tgt_data: Path | list[Path],
    ctx_data: Path | list[Path] | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    ctx_table_name: str | None = None,
    ctx_columns: list[str] | None = None,
    tgt_columns: list[str] | None = None,
    max_sample_size: int | None = None,
    max_sequence_length: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    def get_pqt_files(path) -> list[Path]:
        # convert to list[Path]
        paths = [path] if not isinstance(path, list) else path
        paths = [Path(p) for p in paths]
        # get all parquet files
        files = []
        for path in paths:
            if path.is_dir():
                files += list(path.glob("*.parquet"))
            elif path.is_file() and path.suffix == ".parquet":
                files += [path]
            else:
                pass
        # ensure consistent order by sorting alphabetically
        files = sorted(files)
        return files

    def count_pqt_rows(path) -> int:
        n_rows = 0
        pqt_files = get_pqt_files(path)
        for pqt_file in pqt_files:
            try:
                # read_table returns empty dataframe for dirs with mixed extensions
                n_rows += len(papqt.read_table(pqt_file, columns=[]))
            except (FileNotFoundError, pa.ArrowInvalid):
                continue
        return n_rows

    # read data in memory-safe manner, shuffle, apply max_sample_size, apply max_sequence_length
    if ctx_data is not None:
        # sequential setup
        assert ctx_primary_key is not None
        assert tgt_context_key is not None
        ctx_files = get_pqt_files(ctx_data)
        ctx_keys = pd.concat([pd.read_parquet(f, columns=[ctx_primary_key]) for f in ctx_files]).reset_index(drop=True)
        ctx_keys = ctx_keys.sample(frac=1).head(max_sample_size).reset_index(drop=True)
        ctx_df = pd.concat([pd.read_parquet(f).merge(ctx_keys, on=ctx_primary_key) for f in ctx_files])
        ctx_df = ctx_df.sample(frac=1).reset_index(drop=True)
        ctx_keys = ctx_keys.rename(columns={ctx_primary_key: tgt_context_key})
        tgt_files = get_pqt_files(tgt_data)
        assert len(tgt_files) > 0
        tgt_df = pd.concat([pd.read_parquet(f).merge(ctx_keys, on=tgt_context_key) for f in tgt_files])
        if max_sequence_length is not None:
            tgt_df = tgt_df[tgt_df.groupby(tgt_context_key).cumcount() < max_sequence_length]
        tgt_df = tgt_df.reset_index(drop=True)
    else:
        # flat setup
        ctx_df = None
        tgt_files = get_pqt_files(tgt_data)
        random.shuffle(tgt_files)
        if max_sample_size is None:
            cutoff_idx = len(tgt_files)
        else:
            cum_no_of_rows = accumulate(tgt_files, func=lambda acc, f: acc + count_pqt_rows(f), initial=0)
            cutoff_idx = len(list(takewhile(lambda n_rows: n_rows <= max_sample_size, cum_no_of_rows)))
        tgt_df = pd.concat([pd.read_parquet(f).sample(frac=1) for f in tgt_files[:cutoff_idx]]).head(max_sample_size)
        tgt_df = tgt_df.reset_index(drop=True)

    # dtypes: harmonize string columns
    for col in tgt_df.select_dtypes(include=["object", "string"]).columns:
        tgt_df[col] = tgt_df[col].astype("string[pyarrow]")
    if ctx_df is not None:
        for col in ctx_df.select_dtypes(include=["object", "string"]).columns:
            ctx_df[col] = ctx_df[col].astype("string[pyarrow]")

    # ctx_columns, tgt_columns: pick up relevant columns
    if tgt_columns is not None:
        tgt_columns += [tgt_context_key] if tgt_context_key and tgt_context_key not in tgt_columns else []
        tgt_df = tgt_df[[c for c in tgt_df.columns if c in tgt_columns]]
    if ctx_df is not None and ctx_columns is not None:
        ctx_columns += [ctx_primary_key] if ctx_primary_key and ctx_primary_key not in ctx_columns else []
        ctx_df = ctx_df[[c for c in ctx_df.columns if c in ctx_columns]]

    # since mostlyai-qa expects inputs without prefixes, we have to strip them in advance
    if ctx_df is not None and ctx_table_name:
        ctx_df.columns = strip_column_prefix(prefixed_data=ctx_df.columns, table_name=ctx_table_name)
    return tgt_df, ctx_df
