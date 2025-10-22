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

"""Data pull utilities."""

import itertools
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import xxhash

from mostlyai.sdk._data.base import (
    ContextRelation,
    DataIdentifier,
    NonContextRelation,
    Schema,
)
from mostlyai.sdk._data.context import (
    add_gpc_context,
    add_ns_context,
    add_scp_context,
    drop_unsupported_encoding_types_from_context,
    get_ns_prev_cur_name,
    get_table_chain_to_tgt,
)
from mostlyai.sdk._data.dtype import STRING
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.language_model import (
    drop_language_columns_in_target,
    split_language_model,
)
from mostlyai.sdk._data.non_context import add_is_null_for_non_context_relations
from mostlyai.sdk._data.progress_callback import ProgressCallbackWrapper
from mostlyai.sdk._data.util.common import TEMPORARY_PRIMARY_KEY
from mostlyai.sdk.domain import ModelEncodingType, ModelType

_LOG = logging.getLogger(__name__)


MAX_SAMPLES_PER_ROOT = 5
MAX_TGT_ROWS_PER_CTX_KEY = "__max_tgt_rows_per_ctx_key__"
FRACTION = "fraction"


def determine_n_partitions(
    schema: Schema,
    ctx_nodes: list[str] | None = None,
    tgt_node: str | None = None,
    ctx_n_rows: int | None = None,
    tgt_n_rows: int | None = None,
) -> int:
    ctx_n_rows = ctx_n_rows or 0
    tgt_n_rows = tgt_n_rows or 0
    bytes_per_cell = 8
    # we need to remain conservative here, as we don't know the exact size; particular not for SCP
    max_partition_size = 25 * 1024 * 1024  # 25MB
    ctx_nodes = ctx_nodes or []
    ctx_n_cols = sum(len(schema.tables[node].columns) for node in ctx_nodes if node in schema.tables.keys())
    tgt_n_cols = len(schema.tables[tgt_node].columns) if tgt_node in schema.tables.keys() else 0
    ctx_total_bytes = ctx_n_rows * ctx_n_cols * bytes_per_cell
    tgt_total_bytes = tgt_n_rows * tgt_n_cols * bytes_per_cell
    _LOG.info(
        f"estimated "
        f"ctx_total_bytes: {ctx_total_bytes / 1024**2:.2f}MB, "
        f"tgt_total_bytes: {tgt_total_bytes / 1024**2:.2f}MB"
    )
    total_bytes = ctx_total_bytes + tgt_total_bytes
    n_partitions = max(1, int(np.ceil(total_bytes / max_partition_size)))
    return n_partitions


def _key_columns(
    tgt: str | None = None,
    ctx_tgt_path: list[ContextRelation] = None,
) -> list[str]:
    key_columns = []
    if ctx_tgt_path is not None:
        for rel in ctx_tgt_path:
            key_columns += [rel.parent.ref_name(prefixed=rel.parent.table != tgt)]
            if rel.child.table:
                key_columns += [rel.child.ref_name(prefixed=rel.child.table != tgt)]

    return [i for i in set(key_columns) if i is not None]


def mask_keys(
    key_columns: list[str],
    ctx_data: pd.DataFrame | None = None,
    tgt_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    def mask_column(data: pd.DataFrame):
        for col in key_columns:
            if col in data.columns:
                data[col] = (
                    data[col].apply(lambda x: f"mostly{str(uuid.uuid5(uuid.NAMESPACE_OID, str(x)))[6:]}").astype(STRING)
                )
        return data

    if ctx_data is not None:
        ctx_data = mask_column(ctx_data)

    if tgt_data is not None:
        tgt_data = mask_column(tgt_data)

    return ctx_data, tgt_data


def fetch_ctx_keys(
    schema: Schema,
    ctx_path: list[ContextRelation],
    max_samples_per_root: int | None = None,
    max_sample_size: int | None = None,
) -> pd.DataFrame:
    """Fetch context keys while considering MAX_SAMPLES_PER_ROOT

    :param schema: schema that represents the relevant tables and their relations
    :param ctx_path: a list of ContextRelation representing the path from the root to the context table
    :param max_samples_per_root: restrict ctx_keys per unique root entity; set to None to ignore
    :param max_sample_size: number of rows to sample from the context table, or None for unlimited
    :return: a pd.DataFrame of keys to be fetched
    """
    ctx_keys_identifiers = list(reversed(list(itertools.chain(*[[rel.parent, rel.child] for rel in ctx_path]))))[1:]
    root_key = ctx_keys_identifiers[-1].ref_name()
    tables_keys = {
        table: [g.column for g in list(g)]
        for table, g in itertools.groupby(ctx_keys_identifiers, lambda x: x.table)
        if table
    }

    ctx_keys = pd.DataFrame()
    prev_table_fk = None
    where_values = None
    # traverse ctx_0 -> ctx_1 -> ... -> root
    for table_name, keys in tables_keys.items():
        table = schema.tables[table_name]
        table_pk = keys[0]
        table_fk = keys[1] if len(keys) > 1 else None
        if where_values:
            where = {table_pk: where_values}
        else:
            where = None
        df = table.read_data_prefixed(
            columns=keys,
            where=where,
            do_coerce_dtypes=True,
        ).dropna()
        table_pk_qual = DataIdentifier(table=table_name, column=table_pk).ref_name()
        if table_fk:
            table_fk_qual = DataIdentifier(table=table_name, column=table_fk).ref_name()
            where_values = list(set(df[table_fk_qual]))

        if ctx_keys.empty:
            ctx_keys = df
        else:
            ctx_keys = pd.merge(ctx_keys, df, left_on=prev_table_fk, right_on=table_pk_qual)
        _LOG.info(f"fetched {len(ctx_keys)} ctx_keys")
        prev_table_fk = DataIdentifier(table=table_name, column=table_fk).ref_name()

    # ensure that we do not sample more than MAX_SAMPLES_PER_ROOT
    if max_samples_per_root is not None and len(tables_keys) > 1:
        # shuffle ctx_keys to then filter to first N ctx_keys per root_key
        ctx_keys = ctx_keys.sample(frac=1)
        root_idx = ctx_keys.groupby(root_key).cumcount()
        drop_idx = root_idx >= max_samples_per_root
        if drop_idx.any():
            _LOG.info(f"drop {drop_idx.sum()} ctx_keys to protect privacy at root level")
            ctx_keys = ctx_keys.loc[~drop_idx]
    _LOG.info(f"pulled {len(ctx_keys)} ctx_keys")

    # randomly sample from context keys
    if max_sample_size is not None and max_sample_size < len(ctx_keys):
        _LOG.info(f"randomly sample {max_sample_size} ctx_keys")
        ctx_keys = ctx_keys.sample(n=max_sample_size)
    return ctx_keys


def export_encoding_types(
    schema: Schema,
    columns: list[str],
    tgt_table_name: str | None = None,
) -> dict[str, str]:
    """
    Helper function to collect encoding types for particular columns of a dataframe.
    """
    encoding_types = {}
    for tbl_name, tbl_table in schema.tables.items():
        if tgt_table_name is not None and tbl_name != tgt_table_name:
            # target will not have columns of other tables
            continue
        for col_name, encoding_type in tbl_table.encoding_types.items():
            if col_name in columns:
                encoding_types |= {col_name: encoding_type}
            col_qual_name, prev_col_qual_name = get_ns_prev_cur_name(tbl_name, col_name)
            if col_qual_name in columns:
                encoding_types |= {col_qual_name: encoding_type}
            if prev_col_qual_name in columns:
                encoding_types |= {prev_col_qual_name: encoding_type}  # Nested Sequence
        non_context_relations: list[NonContextRelation] = schema.subset(
            relation_type=NonContextRelation, relations_to=[tbl_name]
        ).relations
        for rel in non_context_relations:
            non_context_is_null = rel.get_is_null_column(is_target=tgt_table_name is not None)
            # check whether non-context relation is present in columns
            if non_context_is_null not in columns:
                continue
            # add non-context `_is_null` column as categorical
            encoding_types |= {non_context_is_null: ModelEncodingType.tabular_categorical}

        # handle potentially unsupported encoding types
        relative_datetime_encodings = {
            k: e for k, e in encoding_types.items() if e == ModelEncodingType.tabular_datetime_relative
        }
        if not schema.get_context_key(tbl_name) and relative_datetime_encodings:
            _LOG.warning(
                f"falling back from TABULAR_DATETIME_RELATIVE to TABULAR_DATETIME for columns {list(relative_datetime_encodings.keys())}"
            )
            fallback_encodings = {k: ModelEncodingType.tabular_datetime for k in relative_datetime_encodings}
            encoding_types |= fallback_encodings

    # convert EncodingTypes to their uppercase string representation
    encoding_types = {k: v.value for k, v in encoding_types.items() if v is not None}
    _LOG.info(f"exported encoding_types: {encoding_types}")
    return encoding_types


def export_meta(
    tgt: str,
    schema: Schema,
    ctx_columns: list[str] | None,
    tgt_columns: list[str] | None,
    model_type: ModelType,
    workspace_dir: Path,
):
    """
    Export the metadata into json files.
    """

    def write_json(data: dict, fn: Path) -> None:
        with open(fn, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)

    # CONTEXT
    tgt_table = schema.tables[tgt]
    ctx = schema.get_parent(tgt) if model_type == ModelType.tabular else tgt
    if ctx:
        ctx_metadata_path = workspace_dir / "OriginalData" / "ctx-meta"
        ctx_metadata_path.mkdir(parents=True, exist_ok=True)
        ctx_pk = schema.tables[ctx].primary_key
        # ctx-meta / keys.json
        if model_type == ModelType.language:
            ctx_keys = {"primary_key": DataIdentifier(tgt, TEMPORARY_PRIMARY_KEY).ref_name()}
        else:
            ctx_keys = {"primary_key": DataIdentifier(ctx, ctx_pk).ref_name()}
        # add root_key, if exists (the exception is single table, 2 models)
        context_tables = schema.get_context_tables(tgt)
        _, ctx_tgt_path = get_table_chain_to_tgt(
            schema=schema,
            tables=context_tables,
            tgt=tgt,
        )
        if ctx_tgt_path[0].parent.column:
            ctx_keys |= {"root_key": ctx_tgt_path[0].parent.ref_name()}
        write_json(ctx_keys, ctx_metadata_path / "keys.json")
        _LOG.info(f"exported context keys.json: {ctx_keys}")
        # ctx-meta / encoding-types.json
        ctx_encoding_types = export_encoding_types(schema, ctx_columns)
        write_json(ctx_encoding_types, ctx_metadata_path / "encoding-types.json")

    # TARGET
    tgt_metadata_path = workspace_dir / "OriginalData" / "tgt-meta"
    tgt_metadata_path.mkdir(parents=True, exist_ok=True)
    # tgt-meta / keys.json
    tgt_pk = pk.column if (pk := schema.get_primary_key(tgt)) else None
    for rel in schema.subset(relations_to=tgt).relations:
        if rel.child == DataIdentifier(tgt, tgt_pk):
            # having the same field as PK and FK is not supported
            # FK is prioritised, thus removing PK role from it
            tgt_pk = None
    if tgt_pk and tgt_pk in tgt_table.columns:
        tgt_keys = {"primary_key": tgt_pk}
    else:
        tgt_keys = {}
    ctx_rel = schema.get_parent_context_relation(tgt)
    if ctx_rel is not None:
        tgt_keys |= {"context_key": ctx_rel.child.column}

    if model_type == ModelType.language:
        tgt_keys = {"context_key": TEMPORARY_PRIMARY_KEY}
    write_json(tgt_keys, tgt_metadata_path / "keys.json")
    _LOG.info(f"exported target keys.json: {tgt_keys}")
    # tgt-meta / encoding-types.json
    tgt_encoding_types = export_encoding_types(schema, tgt_columns, tgt_table_name=tgt)
    write_json(tgt_encoding_types, tgt_metadata_path / "encoding-types.json")


def fetch_tgt_keys(schema: Schema, tgt: str, max_sample_size: int | None = None) -> pd.DataFrame:
    """Fetch target primary keys

    :param schema: schema that represents the relevant tables and their relations
    :param tgt: target table
    :param max_sample_size: number of rows to sample from the target table, or None for unlimited
    :return: a pd.DataFrame of keys to be fetched
    """

    tgt_table = schema.tables[tgt]
    tgt_primary_key = schema.get_primary_key(tgt)
    tgt_keys = tgt_table.read_data_prefixed(columns=[tgt_primary_key.column], do_coerce_dtypes=True)
    _LOG.info(f"pulled {len(tgt_keys)} tgt_keys")

    if max_sample_size is not None and max_sample_size < len(tgt_keys):
        _LOG.info(f"randomly sample {max_sample_size} tgt_keys")
        tgt_keys = tgt_keys.sample(n=max_sample_size)
    return tgt_keys


def handle_workspace_dir(workspace_dir: str | Path | None) -> Path:
    if not workspace_dir:
        workspace_dir = Path()
    else:
        workspace_dir = Path(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def pull_keys(
    *,
    tgt: str,
    schema: Schema,
    max_sample_size: int | None,
    model_type: ModelType = ModelType.tabular,
) -> pd.DataFrame | None:
    """Pull target or context keys.

    :param tgt: name of the target data table
    :param schema: Schema object
    :param max_sample_size: number of rows to sample from the target / context table, or None for unlimited
    :param model_type: model type for the target data

    :return: DataFrame containing keys to be fetched
    """

    _LOG.info("HELLO FROM PULL_KEYS")
    t0 = time.time()
    context_tables = schema.get_context_tables(tgt)
    ctx_tgt_nodes, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    ctx_path = ctx_tgt_path[:-1]

    # determine keys to fetch
    ctx_keys = tgt_keys = None
    if ctx_path:
        # fetch context keys for sequential setup
        ctx_keys = fetch_ctx_keys(
            schema=schema,
            ctx_path=ctx_path,
            max_samples_per_root=MAX_SAMPLES_PER_ROOT,
            max_sample_size=max_sample_size,
        )
    elif schema.get_primary_key(tgt) is not None:
        # fetch target primary keys for flat setup
        tgt_keys = fetch_tgt_keys(schema=schema, tgt=tgt, max_sample_size=max_sample_size)
    keys = ctx_keys if ctx_keys is not None else tgt_keys
    if model_type == ModelType.language and ctx_keys is not None:
        keys = add_max_tgt_rows_per_ctx_key(ctx_tgt_path, keys, schema, tgt)
    _LOG.info(f"BYE FROM PULL_KEYS (total time: {time.time() - t0:.2f}s)")
    return keys


def add_max_tgt_rows_per_ctx_key(ctx_tgt_path: list[ContextRelation], keys: pd.DataFrame, schema: Schema, tgt: str):
    """
    Calculates and adds the maximum target rows per context key to the provided keys DataFrame.
    That is to respect MAX_SAMPLES_PER_ROOT at a later stage, when some of the tgt columns end up being part of ctx.
    """
    tgt_table = schema.tables[tgt]
    ctx_tgt_relation = schema.get_parent_context_relation(tgt)
    ctx_tgt_keys_counts = (
        tgt_table.read_data_prefixed(columns=[ctx_tgt_relation.child.column], do_coerce_dtypes=True)
        .squeeze(axis=1)
        .value_counts()
        .reset_index()
    )
    keys = pd.merge(
        keys,
        ctx_tgt_keys_counts,
        left_on=ctx_tgt_relation.parent.ref_name(),
        right_on=ctx_tgt_relation.child.ref_name(),
    )
    root_key = ctx_tgt_path[0].parent.ref_name()
    keys["total_count"] = keys.groupby(root_key)["count"].transform("sum")
    keys[MAX_TGT_ROWS_PER_CTX_KEY] = MAX_SAMPLES_PER_ROOT * (keys["count"] / keys["total_count"])
    keys = keys.drop(columns=["count", "total_count"])
    return keys


def fetch_table_data(
    schema: Schema,
    table_name: str,
    fetch_dir: Path,
    deduplicate_pks: bool,
    where: dict | None,
    sample_fraction: float | None,
    key_fraction_df: pd.DataFrame | None,
    progress: ProgressCallbackWrapper,
):
    """Fetch table data.

    :param schema: Schema object
    :param table_name: name of the table to fetch
    :param fetch_dir: directory to save fetched data
    :param deduplicate_pks: whether to deduplicate primary keys,
        this is used when fetching context primary keys and we need to ensure uniqueness
    :param where: where clause to filter the data
    :param sample_fraction: fraction of rows to keep from the table; None to keep all rows
    :param key_fraction_df: a pd.DataFrame with two columns: key (its name in tgt table) and a
        fraction of overall values to fetch (grouped by the given corresponding key)
    :param progress: callback to report progress
    """

    t0 = time.time()
    table = schema.tables[table_name]
    primary_key = schema.get_primary_key(table_name)
    keys = set()
    iterator = table.read_chunks(
        where=where,
        columns=table.columns,
        do_coerce_dtypes=True,
        yield_chunk_size=1_000_000,
    )
    n_fetched_rows = 0
    for idx, chunk_df in enumerate(iterator):
        if deduplicate_pks and primary_key and primary_key.column in chunk_df.columns:
            # consider only the first occurrence of each primary key
            if not chunk_df[primary_key.column].is_unique:
                drop_idx = chunk_df[primary_key.column].duplicated()
                _LOG.warning(
                    f"drop {drop_idx.sum()} records due to duplicate primary keys in `{primary_key.ref_name()}`"
                )
                chunk_df = chunk_df.loc[~drop_idx]
            keys.update(chunk_df[primary_key.column])
        if key_fraction_df is not None:
            # the number of rows as a float
            chunk_size = len(chunk_df)
            key_fraction_df["n_rows"] = key_fraction_df[FRACTION] * chunk_size

            # the integer part and the fractional part
            int_part = key_fraction_df["n_rows"].astype(int)
            frac_part = key_fraction_df["n_rows"] - int_part

            # a random number for each row to decide if we round up
            random_draws = np.random.rand(len(key_fraction_df))

            # n_rows with rounding based on the fraction
            key_fraction_df["n_rows"] = int_part + (random_draws < frac_part).astype(int)

            ctx_key = key_fraction_df.columns[0]
            chunk_df = chunk_df.sample(frac=1)

            def sample_n_rows(group):
                n = key_fraction_df.loc[key_fraction_df[ctx_key] == group.name, "n_rows"].values[0]
                return group.sample(n=min(n, len(group)))  # don't exceed available rows

            # group by ctx_key and apply the sampling function
            chunk_df = chunk_df.groupby(ctx_key, group_keys=False).apply(sample_n_rows).reset_index(drop=True)

            _LOG.info(f"drop {chunk_size - len(chunk_df)} ctx_keys to protect privacy")
        if sample_fraction is not None:
            no_of_keep_rows = int(sample_fraction * len(chunk_df))
            keep_idx = [True] * no_of_keep_rows + [False] * (len(chunk_df) - no_of_keep_rows)
            np.random.shuffle(keep_idx)
            chunk_df = chunk_df.iloc[keep_idx]
        chunk_path = fetch_dir / table_name / f"chunk.{idx:06}.parquet"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_df.to_parquet(chunk_path, index=False)
        n_fetched_rows += len(chunk_df)
        # increment progress by the number of rows in the chunk
        progress.update(advance=len(chunk_df))
    # ensure that we ultimately incremented by the total number of rows
    progress.update(advance=table.row_count - n_fetched_rows)
    _LOG.info(f"table {table_name} fetched in {time.time() - t0:.2f}s")


def fetch_context_tables(
    schema: Schema,
    tgt: str,
    keys: pd.DataFrame | None,
    fetch_dir: Path,
    progress: ProgressCallbackWrapper,
):
    context_tables = schema.get_context_tables(tgt)
    ctx_tgt_nodes, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    for parent_table, child_table in zip(ctx_tgt_nodes[:-1], ctx_tgt_nodes[1:]):
        # fetch parent table context
        primary_key = schema.get_primary_key(parent_table)
        pks = keys[primary_key.ref_name()].unique()
        fetch_table_data(
            schema=schema,
            table_name=parent_table,
            fetch_dir=fetch_dir,
            deduplicate_pks=True,  # drop duplicate PKs in parent table context
            where={primary_key.column: pks},
            sample_fraction=None,
            key_fraction_df=None,
            progress=progress,
        )

        # fetch cross table contexts
        sibling_tables = schema.get_older_sibling_tables(parent=parent_table, child=child_table)
        for sibling_table in sibling_tables:
            context_key = schema.get_context_key(sibling_table)
            fetch_table_data(
                schema=schema,
                table_name=sibling_table,
                fetch_dir=fetch_dir,
                deduplicate_pks=True,  # drop duplicate PKs in cross table context
                where={context_key.column: pks},
                sample_fraction=None,
                key_fraction_df=None,
                progress=progress,
            )


def fetch_target_table(
    schema: Schema,
    tgt: str,
    keys: pd.DataFrame | None,
    max_sample_size: int | None,
    fetch_dir: Path,
    progress: ProgressCallbackWrapper,
):
    ctx = schema.get_parent(tgt)
    tgt_primary_key = schema.get_primary_key(tgt)
    sample_fraction = None
    key_fraction_df = None
    if ctx is not None:
        # sequential setup
        # sample target table by context primary key
        # use provided keys to filter and order the target table (max_sample_size=None, do_shuffle=False)
        ctx_primary_key = schema.get_primary_key(ctx)
        tgt_context_key = schema.get_context_key(tgt)
        pks = keys[ctx_primary_key.ref_name()].unique()
        where = {tgt_context_key.column: pks}
        if MAX_TGT_ROWS_PER_CTX_KEY in keys.columns:
            key_fraction_df = keys[[tgt_context_key.ref_name(), MAX_TGT_ROWS_PER_CTX_KEY]]
            key_fraction_df[FRACTION] = keys[MAX_TGT_ROWS_PER_CTX_KEY] / schema.tables[tgt].row_count
            key_fraction_df = key_fraction_df.rename(columns={tgt_context_key.ref_name(): tgt_context_key.column})
            key_fraction_df = key_fraction_df.drop(columns=[MAX_TGT_ROWS_PER_CTX_KEY])
    elif tgt_primary_key:
        # flat setup with primary key
        # sample target table by target primary key
        # use provided keys to filter and order the target table (max_sample_size=None, do_shuffle=False)
        pks = keys[tgt_primary_key.ref_name()].unique()
        where = {tgt_primary_key.column: pks}
    else:
        # flat setup without primary key
        # sample target table by taking first max_sample_size rows (or all rows)
        # shuffle the target table (do_shuffle=True)
        where = None
        if max_sample_size is not None and max_sample_size > 0:
            sample_fraction = min(1.0, max_sample_size / schema.tables[tgt].row_count)
    fetch_table_data(
        schema=schema,
        table_name=tgt,
        fetch_dir=fetch_dir,
        deduplicate_pks=True,  # drop duplicate PKs in target table
        where=where,
        key_fraction_df=key_fraction_df,
        sample_fraction=sample_fraction,
        progress=progress,
    )


def pull_fetch(
    *,
    tgt: str,
    schema: Schema,
    keys: pd.DataFrame | None,
    max_sample_size: int | None,
    workspace_dir: Path,
    progress: ProgressCallbackWrapper,
) -> None:
    """Fetch target and context tables to `workspace_dir / __PULL_FETCH`.

    :param tgt: name of the target data table
    :param schema: Schema object
    :param keys: DataFrame containing keys to pull
    :param max_sample_size: used for flat setup without primary key,
        in which case first max_sample_size rows are fetched
    :param workspace_dir: workspace directory
    :param progress: callback to report progress
    """

    _LOG.info("HELLO FROM PULL_FETCH")
    t0 = time.time()
    fetch_dir = workspace_dir / "__PULL_FETCH"
    shutil.rmtree(fetch_dir, ignore_errors=True)
    fetch_dir.mkdir(exist_ok=True, parents=True)
    fetch_context_tables(schema=schema, tgt=tgt, keys=keys, fetch_dir=fetch_dir, progress=progress)
    fetch_target_table(
        schema=schema,
        tgt=tgt,
        keys=keys,
        max_sample_size=max_sample_size,
        fetch_dir=fetch_dir,
        progress=progress,
    )
    _LOG.info(f"BYE FROM PULL_FETCH (total time: {time.time() - t0:.2f}s)")


def remake_schema_after_pull_fetch(tgt: str, schema: Schema, workspace_dir: Path) -> Schema:
    tables = {}
    fetched_tables = schema.get_context_tables(tgt) + [tgt]
    for table_name, table in schema.tables.items():
        tables[table_name] = table
        if table_name in fetched_tables:
            fs_table = ParquetDataTable(path=workspace_dir / "__PULL_FETCH" / table_name, name=table_name)
            fs_table.primary_key = table.primary_key
            fs_table.foreign_keys = table.foreign_keys
            fs_table.encoding_types = table.encoding_types
            tables[table_name] = fs_table
    schema = Schema(tables=tables)
    schema.preprocess_schema_before_pull()
    return schema


def export_chunk(
    chunk_idx: int,
    chunk: pd.DataFrame,
    hash_column: pd.Series,
    n_partitions: int,
    trn_val_split: float | None,
    data_dir: Path,
    do_ctx_only: bool,
):
    """Distributes rows of a chunk into partitions using hash trick."""

    data_dir.mkdir(exist_ok=True, parents=True)

    def _hash_partitioner(x: pd.Series, n_partitions: int) -> np.array:
        hasher = np.vectorize(lambda s: xxhash.xxh32_intdigest(str(s)) % n_partitions, otypes=[int])
        return hasher(x)

    def _store_partition_chunk(partition_idx: str, partition_chunk: pd.DataFrame):
        partition_dir = data_dir / ("part." + partition_idx)
        partition_dir.mkdir(exist_ok=True, parents=True)
        partition_chunk_path = partition_dir / f"chunk.{chunk_idx:06}.parquet"
        (
            partition_chunk.drop(columns=["__PARTITION_GROUP", "__PARTITION_SPLIT"], errors="ignore")
            .reset_index(drop=True)
            .to_parquet(partition_chunk_path, index=True)
        )

    if len(chunk) == 0:
        for split in ["trn", "val"] if not do_ctx_only else ["ctx"]:
            _store_partition_chunk(partition_idx=f"000000-{split}", partition_chunk=chunk)
        return

    # split into partitions; plus split each partition into trn/val based on trn_val_split
    # for that we create 10x more partitions, map modulo based on trn_val_split
    # don't split into partitions when pulling context only
    trn_val_split = trn_val_split or 0.8
    val_count = round((1 - trn_val_split) * 10)  # number of partitions for validation (out of 10)
    hashes = _hash_partitioner(hash_column, 10 * n_partitions)
    if np.all(hashes % 10 < val_count):
        hashes += val_count  # ensure that we have at least one training partition

    chunk["__PARTITION_GROUP"] = hashes // 10
    chunk["__PARTITION_SPLIT"] = np.where(hashes % 10 < val_count, "val", "trn") if not do_ctx_only else "ctx"

    for partition_group in chunk["__PARTITION_GROUP"].unique():
        partition_chunk = chunk[chunk["__PARTITION_GROUP"] == partition_group]
        for partition_split in partition_chunk["__PARTITION_SPLIT"].unique():
            partition_chunk_split = partition_chunk[partition_chunk["__PARTITION_SPLIT"] == partition_split]
            _store_partition_chunk(
                partition_idx=f"{partition_group:06d}-{partition_split}",
                partition_chunk=partition_chunk_split,
            )


def consolidate_partitions(data_dir: Path, shuffle: bool = True):
    """Consolidates partition chunks into partitions."""
    t0 = time.time()
    for partition_dir in sorted(d for d in data_dir.glob("part.*/") if d.is_dir()):
        partition_data = pd.concat(
            (pd.read_parquet(chunk_path) for chunk_path in sorted(partition_dir.iterdir())),
            ignore_index=True,
        )
        if shuffle:
            partition_data = partition_data.sample(frac=1)
        shutil.rmtree(partition_dir)
        partition_path = data_dir / f"{partition_dir.name}.parquet"
        partition_data.reset_index(drop=True).to_parquet(partition_path, index=True)
    _LOG.info(f"consolidated chunks into partitions in {time.time() - t0:.2f}s")


def fill_missing_tgt_partitions(ctx_data_dir: Path, tgt_data_dir: Path, tgt_columns: list[str]):
    """
    Ensures that each context partition has a corresponding target partition
    by filling the missing partitions with empty data
    """
    for ctx_partition_dir in ctx_data_dir.glob("part.*.parquet"):
        tgt_partition_path = tgt_data_dir / ctx_partition_dir.name
        if not tgt_partition_path.exists():
            tgt_partition_path.parent.mkdir(exist_ok=True, parents=True)
            tgt_partition = pd.DataFrame(columns=tgt_columns)
            tgt_partition.to_parquet(tgt_partition_path, index=False)


def split_context(
    tgt: str,
    schema: Schema,
    ctx_data_dir: Path,
    n_partitions: int,
    trn_val_split: float | None,
    model_type: ModelType,
    do_ctx_only: bool,
    progress: ProgressCallbackWrapper,
):
    ctx = schema.get_parent(tgt)
    context_tables = schema.get_context_tables(tgt)
    _, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    if ctx is not None:
        t0 = time.time()
        ctx_table = schema.tables[ctx] if ctx else None
        iterator = ctx_table.read_chunks_prefixed(do_coerce_dtypes=True, fetch_chunk_size=100_000)
        table = ctx_table
        key = schema.get_primary_key(table.name)
        for idx, chunk in enumerate(iterator):
            # add GPC context
            chunk = add_gpc_context(
                chunk=chunk,
                schema=schema,
                tgt=tgt,
            )
            if idx == 0:
                _LOG.info(f"{chunk.shape=} (post adding GPC context)")

            if model_type == ModelType.language:
                # remove non-context columns
                non_ctx_cols = [
                    # when pulling data for a GENERATION job (do_ctx_only=True), these columns have a special suffix
                    rel.get_is_null_column(is_target=False) if do_ctx_only else rel.child.ref_name()
                    for rel in schema.subset(
                        relation_type=NonContextRelation,
                        relations_to=[table.name],
                    ).relations
                ]
                chunk = chunk.drop(columns=non_ctx_cols)
            else:
                chunk = add_is_null_for_non_context_relations(
                    schema=schema,
                    table_name=table.name,
                    data=chunk,
                    is_target=False,
                )
                if idx == 0:
                    _LOG.info(f"{chunk.shape=} (post handle non-context relations)")

                # add SCP context
                chunk = add_scp_context(
                    tgt=tgt,
                    schema=schema,
                    ctx_keys=chunk,
                    ctx_data=chunk,
                    do_coerce_dtypes=True,
                )
                if idx == 0:
                    _LOG.info(f"{chunk.shape=} (post adding SCP context)")

                # add NS context
                chunk = add_ns_context(
                    tgt=tgt,
                    schema=schema,
                    ctx_data=chunk,
                )
                if idx == 0:
                    _LOG.info(f"{chunk.shape=} (post adding NS context)")

            # drop unsupported column types from context
            chunk = drop_unsupported_encoding_types_from_context(
                tgt=tgt,
                schema=schema,
                ctx_data=chunk,
            )
            if idx == 0:
                _LOG.info(f"{chunk.shape=} (post removing unsupported context encoding types)")

            # mask chunk keys (only when pulling training data)
            if not do_ctx_only:
                key_columns = _key_columns(ctx_tgt_path=ctx_tgt_path)
                chunk, _ = mask_keys(key_columns=key_columns, ctx_data=chunk)
                if idx == 0:
                    _LOG.info(f"{chunk.shape=} (post masking keys)")

            # make partition chunks
            export_chunk(
                chunk_idx=idx,
                chunk=chunk,
                hash_column=chunk[key.ref_name()],
                n_partitions=n_partitions,
                trn_val_split=trn_val_split,
                data_dir=ctx_data_dir,
                do_ctx_only=do_ctx_only,
            )
            progress.update(advance=len(chunk))
        consolidate_partitions(
            ctx_data_dir,
            # DO shuffle when pulling training data
            # DON'T shuffle when pulling generation context
            shuffle=not do_ctx_only,
        )
        _LOG.info(f"ctx partitions created in {time.time() - t0:.2f}s")


def split_target(
    tgt: str,
    schema: Schema,
    tgt_data_dir: Path,
    n_partitions: int,
    trn_val_split: float | None,
    model_type: ModelType,
    do_ctx_only: bool,
    progress: ProgressCallbackWrapper,
):
    tgt_table = schema.tables[tgt]
    ctx = schema.get_parent(tgt)
    context_tables = schema.get_context_tables(tgt)
    ctx_tgt_nodes, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    if model_type == ModelType.language or not do_ctx_only:
        t0 = time.time()
        iterator = tgt_table.read_chunks_prefixed(fetch_chunk_size=1_000_000, include_table_prefix=False)
        table = tgt_table

        def _hash_column(chunk):
            if ctx is not None:
                key = schema.get_context_key(table.name).column
                return chunk[key]
            return chunk.index

        for idx, chunk in enumerate(iterator):
            chunk = add_is_null_for_non_context_relations(
                schema=schema,
                table_name=table.name,
                data=chunk,
                is_target=True,
            )
            if idx == 0:
                _LOG.info(f"{chunk.shape=} (post handle non-context relations)")

            if model_type == ModelType.tabular:
                # drop LANGUAGE columns
                chunk = drop_language_columns_in_target(
                    tgt=tgt,
                    schema=schema,
                    tgt_data=chunk,
                )
            if idx == 0:
                _LOG.info(f"{chunk.shape=} (post optional removal of LANGUAGE columns)")

            # mask chunk keys (only when pulling training data)
            if not do_ctx_only:
                key_columns = _key_columns(ctx_tgt_path=ctx_tgt_path, tgt=tgt)
                _, chunk = mask_keys(key_columns=key_columns, tgt_data=chunk)
                if idx == 0:
                    _LOG.info(f"{chunk.shape=} (post masking keys)")

            # make partition chunks
            export_chunk(
                chunk_idx=idx,
                chunk=chunk,
                hash_column=_hash_column(chunk),
                n_partitions=n_partitions,
                trn_val_split=trn_val_split,
                data_dir=tgt_data_dir,
                do_ctx_only=do_ctx_only,
            )
            progress.update(advance=len(chunk))
        consolidate_partitions(
            tgt_data_dir,
            # DO shuffle when pulling training data for flat setup
            # DON'T shuffle when pulling training data for sequential setup
            # DON'T shuffle when pulling generation context
            shuffle=not (do_ctx_only or context_tables),
        )
        _LOG.info(f"tgt partitions created in {time.time() - t0:.2f}s")


def repartition_language_model(tgt: str, schema: Schema, ctx_data_dir: Path, tgt_data_dir: Path):
    t0 = time.time()
    tgt_partition_paths = sorted(list(tgt_data_dir.glob("part.*.parquet")))
    for tgt_partition_path in tgt_partition_paths:
        # read tgt and ctx partitions together
        tgt_df = pd.read_parquet(tgt_partition_path)
        ctx_df = None
        ctx_partition_path = ctx_data_dir / tgt_partition_path.name
        if ctx_partition_path.exists():
            ctx_df = pd.read_parquet(ctx_partition_path)

        # push TABULAR columns to context
        ctx_df, tgt_df = split_language_model(
            schema=schema,
            tgt=tgt,
            tgt_data=tgt_df,
            ctx_data=ctx_df,
        )

        # export tgt and ctx partitions
        tgt_df.reset_index(drop=True).to_parquet(tgt_partition_path, index=True)
        if ctx_df is not None:
            ctx_partition_path.parent.mkdir(exist_ok=True, parents=True)
            ctx_df.reset_index(drop=True).to_parquet(ctx_partition_path, index=True)
    _LOG.info(f"LANGUAGE columns handled in {time.time() - t0:.2f}s")


def pull_split(
    *,
    tgt: str,
    schema: Schema,
    trn_val_split: float | None,
    model_type: ModelType,
    do_ctx_only: bool,
    workspace_dir: Path,
    progress: ProgressCallbackWrapper,
) -> None:
    """Split fetched data among partitions.

    :param tgt: name of the target data table
    :param schema: Schema object
    :param workspace_dir: workspace directory
    :param do_ctx_only: indicates whether context only should be handled
    :param model_type: model type for the target data
    :param progress: callback to report progress
    """

    _LOG.info("HELLO FROM PULL_SPLIT")
    t00 = time.time()
    ctx = schema.get_parent(tgt)
    tgt_table = schema.tables[tgt]
    ctx_table = schema.tables[ctx] if ctx else None
    context_tables = schema.get_context_tables(tgt)
    ctx_tgt_nodes, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    tgt_data_dir = workspace_dir / "OriginalData" / "tgt-data"
    ctx_data_dir = workspace_dir / "OriginalData" / "ctx-data"

    # determine number of training partitions
    _LOG.info(f"{tgt_table=} {ctx_table=}")
    tgt_n_rows = tgt_table.row_count
    ctx_n_rows = ctx_table.row_count if ctx_table else None
    _LOG.info(f"{tgt_n_rows=} {ctx_n_rows=}")
    n_partitions = determine_n_partitions(
        schema=schema,
        ctx_nodes=ctx_tgt_nodes[:-1],
        tgt_node=tgt,
        ctx_n_rows=ctx_n_rows,
        tgt_n_rows=tgt_n_rows,
    )
    _LOG.info(f"{n_partitions=}")

    # split context data
    split_context(
        tgt=tgt,
        schema=schema,
        ctx_data_dir=ctx_data_dir,
        n_partitions=n_partitions,
        trn_val_split=trn_val_split,
        model_type=model_type,
        do_ctx_only=do_ctx_only,
        progress=progress,
    )

    # split target data
    split_target(
        tgt=tgt,
        schema=schema,
        tgt_data_dir=tgt_data_dir,
        n_partitions=n_partitions,
        trn_val_split=trn_val_split,
        model_type=model_type,
        do_ctx_only=do_ctx_only,
        progress=progress,
    )

    # fill missing target partitions in case context partition has 0-seqlens only
    tgt_columns = ParquetDataTable(path=tgt_data_dir).get_columns(exclude_complex_types=False)
    fill_missing_tgt_partitions(ctx_data_dir=ctx_data_dir, tgt_data_dir=tgt_data_dir, tgt_columns=tgt_columns)

    # split LANGUAGE columns
    # this step must come after split_target and split_context
    if model_type == ModelType.language:
        repartition_language_model(
            tgt=tgt,
            schema=schema,
            ctx_data_dir=ctx_data_dir,
            tgt_data_dir=tgt_data_dir,
        )

    if do_ctx_only:
        # ensure target partitions are removed in case of context only
        shutil.rmtree(tgt_data_dir, ignore_errors=True)

    if not do_ctx_only:
        # export meta only in case of pull for training
        t0 = time.time()
        ctx_columns = (
            ParquetDataTable(path=ctx_data_dir).get_columns(exclude_complex_types=False)
            if ctx_data_dir.exists()
            else None
        )
        tgt_columns = ParquetDataTable(path=tgt_data_dir).get_columns(exclude_complex_types=False)
        export_meta(
            tgt=tgt,
            schema=schema,
            ctx_columns=ctx_columns,
            tgt_columns=tgt_columns,
            model_type=model_type,
            workspace_dir=workspace_dir,
        )
        _LOG.info(f"meta exported in {time.time() - t0:.2f}s")
    _LOG.info(f"BYE FROM PULL_SPLIT (total time: {time.time() - t00:.2f}s)")
