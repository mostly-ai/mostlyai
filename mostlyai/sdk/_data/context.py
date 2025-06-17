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

import numpy as np
import pandas as pd

from mostlyai.sdk._data.base import ContextRelation, DataIdentifier, Schema
from mostlyai.sdk.domain import ModelEncodingType, ModelType

_LOG = logging.getLogger(__name__)

NES_SEQ_PREV = "$prev"
MAX_SCP_SEQLEN_LIMIT = 1_000
MAX_NS_PREV_LEN = 20


def add_gpc_context(
    chunk: pd.DataFrame,
    schema: Schema,
    tgt: str,
) -> pd.DataFrame:
    context_tables = schema.get_context_tables(tgt)
    ctx_tgt_nodes, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    ctx_nodes = list(reversed(ctx_tgt_nodes[:-1]))
    for grandparent, parent in zip(ctx_nodes[1:], ctx_nodes[:-1]):
        grandparent_table = schema.tables[grandparent]
        grandparent_primary_key = schema.get_primary_key(grandparent)
        parent_context_key = schema.get_context_key(parent)
        grandparent_data = grandparent_table.read_data_prefixed(
            columns=grandparent_table.columns,
            where={grandparent_primary_key.column: chunk[parent_context_key.ref_name()]},
            do_coerce_dtypes=True,
        )
        _LOG.info(f"fetched {len(grandparent_data)} GPC records")
        chunk = pd.merge(
            chunk,
            grandparent_data,
            left_on=parent_context_key.ref_name(),
            right_on=grandparent_primary_key.ref_name(),
        )
    return chunk


def add_ns_context(
    tgt: str,
    schema: Schema,
    ctx_data: pd.DataFrame,
) -> pd.DataFrame:
    context_tables = schema.get_context_tables(tgt)
    ctx_tgt_nodes, ctx_tgt_path = get_table_chain_to_tgt(
        schema=schema,
        tables=context_tables,
        tgt=tgt,
    )
    if len(ctx_tgt_path) < 3:
        # below 3 tables (S -> L1 -> L2), there is no value in adding NS
        return ctx_data.sample(frac=1)

    root_key = ctx_tgt_path[0].parent.ref_name()
    tgt_table_name = ctx_tgt_path[-1].parent.table
    ns_columns = []

    # first, process nested sequences based on ctx_data
    for ctx_rel in ctx_tgt_path:
        cur_table_name = ctx_rel.child.table
        if not cur_table_name or cur_table_name == tgt_table_name:
            continue

        cur_table = schema.tables[cur_table_name]
        non_key_columns = [k for k, v in cur_table.encoding_types.items()]
        # de-prefix column names, if required
        prefix = DataIdentifier(cur_table_name).ref_name()
        is_prefixed = all(c.startswith(prefix) for c in non_key_columns)
        raw_column_start = len(prefix) if is_prefixed else 0
        for cur_col_name in non_key_columns:
            cur_name, prev_name = get_ns_prev_cur_name(cur_table_name, cur_col_name, raw_column_start)
            cur_dtype = ctx_data.dtypes[cur_name].type

            def add_previous_values_ctx(group):
                # due to lack of NAs support for int*, float*, bool in numpy, fillna was added
                group_ser = (
                    group[cur_name].fillna(0)
                    if any(dtype in str(cur_dtype) for dtype in ["int", "float", "bool"])
                    else group[cur_name]
                )
                group[prev_name] = [np.array(group_ser.values[:i], dtype=cur_dtype) for i in range(len(group))]
                return group

            # create list of the previous values and insert that column adjacent to the current column
            _LOG.info(f"max_ns_prev_len: {MAX_NS_PREV_LEN}")
            prev_ser = (
                ctx_data[[root_key, cur_name]].groupby(root_key).apply(add_previous_values_ctx)[prev_name].droplevel(0)
            ).apply(lambda x: x[:-MAX_NS_PREV_LEN])
            ctx_data.insert(loc=ctx_data.columns.get_loc(cur_name), column=prev_name, value=prev_ser)
            ns_columns.append(prev_name)
    _LOG.info(f"NS was applied, adding the following columns: {ns_columns}")
    return _shuffle_groups(ctx_data, root_key)


def get_ns_prev_cur_name(table_name: str, column_name: str, raw_column_start: int = 0) -> tuple[str, str]:
    # qualified names of the columns to represent: array of previous entries and the current one
    table_name_prev = f"{table_name}{NES_SEQ_PREV}"
    prev_name = DataIdentifier(table_name_prev, column_name[raw_column_start:]).ref_name()
    cur_name = DataIdentifier(table_name, column_name).ref_name()
    return cur_name, prev_name


def _shuffle_groups(df: pd.DataFrame, root_key: str) -> pd.DataFrame:
    """
    Shuffle a dataframe group-element-count-wise by a given root key.
    E.g. shuffled rows of element 0 of all root keys will be in the first batch of rows,
    shuffled rows of element 1 of all root keys will be consequent batch of rows, etc.
    This guarantees preservation of order for each root key within the full dataframe.
    """
    group_order = "__group_order__"
    # root_key is the column to group by
    grouped = df.groupby(root_key)
    # Adding a column 'group_order' that specifies the order within each group
    df[group_order] = grouped.cumcount()
    # Shuffle the index
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    # Sorting by group_order within each group
    shuffled_df.sort_values(by=[group_order], inplace=True)
    # Remove the temporary group_order column
    shuffled_df.drop(columns=group_order, inplace=True)
    return shuffled_df


def add_scp_context(
    schema: Schema,
    tgt: str,
    ctx_keys: pd.DataFrame,
    ctx_data: pd.DataFrame,
    do_coerce_dtypes: bool,
):
    context_tables = schema.get_context_tables(tgt)
    seq_ctx_rels = get_scp_relations(schema, tgt, context_tables)

    if not seq_ctx_rels:
        return ctx_data

    for seq_ctx_rel in seq_ctx_rels:
        parent_key_prefixed = seq_ctx_rel.parent.ref_name()
        child_key_prefixed = seq_ctx_rel.child.ref_name()
        child_key = seq_ctx_rel.child.column
        scp_table = schema.tables[seq_ctx_rel.child.table]
        df_sibling = scp_table.read_data_prefixed(
            columns=scp_table.columns,
            where={child_key: (ctx_keys[parent_key_prefixed])},
            do_coerce_dtypes=do_coerce_dtypes,
            shuffle=False,
        )

        # filtering sibling data with the keys presented on tgt
        df_ids = ctx_keys[[parent_key_prefixed]].drop_duplicates()
        df_sibling = pd.merge(
            df_ids,
            df_sibling,
            left_on=parent_key_prefixed,
            right_on=child_key_prefixed,
            how="left",
            indicator=True,  # useful to map empty sequences
        ).set_index(parent_key_prefixed)

        empty_seqs = df_sibling[df_sibling["_merge"] == "left_only"].index
        df_sibling = df_sibling.drop(columns=["_merge"])

        # grouping rows in order to create the array cells
        grouped = df_sibling.groupby(parent_key_prefixed)
        seq_ctx_df = [grouped[prefixed_col].apply(np.array) for prefixed_col in df_sibling]
        seq_ctx_df = pd.concat(seq_ctx_df, axis=1)

        # replacing empty sequences for empty arrays, instead of [NaN]
        seq_ctx_df.loc[empty_seqs] = seq_ctx_df.loc[empty_seqs].map(lambda x: np.array([], dtype=x.dtype))

        # apply limit to SCP length
        _LOG.info(f"max_scp_sequence_length: {MAX_SCP_SEQLEN_LIMIT}")
        seq_ctx_df = seq_ctx_df.map(lambda x: x[:MAX_SCP_SEQLEN_LIMIT])

        # IMPROVE: instead of merging, can we update ctx_data directly?
        ctx_data = ctx_data.merge(seq_ctx_df, how="left", on=parent_key_prefixed)

    return ctx_data


def get_scp_relations(schema: Schema, table_name: str, context_tables: list[str]) -> list[ContextRelation]:
    rels = schema.get_scp_relations(table_name)

    context_tables = context_tables or []
    rels = list(filter(lambda x: x.child.table in context_tables, rels))

    if len(rels) > 0:
        siblings = [x.child.table for x in rels]
        _LOG.info(f"context siblings (size: {len(rels)}): {siblings}")

    return rels


def drop_unsupported_encoding_types_from_context(
    tgt: str,
    schema: Schema,
    ctx_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop unsupported encoding types from context.

    :param tgt: target table
    :param schema: database schema.
    :param ctx_data: the context DataFrame.
    :return: ctx_data
    """
    context_tables = schema.get_context_tables(tgt)
    sub_schema = schema.subset(tables=context_tables)
    unsupported_encoding_types = [ModelEncodingType.tabular_datetime_relative] + [
        t for t in ModelEncodingType if t.value.startswith(ModelType.language.value)
    ]
    # do not drop context primary key of parent context table
    ctx_pk = sub_schema.tables[context_tables[-1]].primary_key
    drop_columns = []
    for table_name, table in sub_schema.tables.items():
        for col_name, encoding_type in table.encoding_types.items():
            if col_name != ctx_pk and encoding_type in unsupported_encoding_types:
                col_qual_name, prev_col_qual_name = get_ns_prev_cur_name(table_name, col_name)
                drop_columns.append(col_qual_name)
                if prev_col_qual_name in ctx_data:
                    drop_columns.append(prev_col_qual_name)
    if drop_columns:
        _LOG.info(f"drop unsupported encoding_types from context: {drop_columns}")
    return ctx_data.drop(columns=drop_columns, errors="ignore")


def get_table_chain_to_tgt(schema: Schema, tables: list[str], tgt: str) -> tuple[list[str], list[ContextRelation]]:
    """Chain tables to a given tgt via context relations using a given list of tables to consider.
    :param schema: schema that represents the relevant tables and their relations
    :param tables: list of tables to be considered from the schema, e.g. [ctx_0, ctx_1, ..., root]
    :param tgt: tgt table name
    :return: a tuple of nodes (list[str]) and a path (list[ContextRelation]) ending with (tgt.pk <-> None) relation,
    e.g. [root.pk, ..., ctx_0_fk, ctx_0_pk, tgt_fk, tgt_pk],
    [(root.pk <-> ctx_-1_fk), ..., (ctx_0_pk <-> tgt_fk), (tgt_pk <-> None)]
    """
    sub_schema = schema.subset(relation_type=ContextRelation, tables=tables + [tgt])
    nodes, path = sub_schema.path_to(tgt)
    tgt_pk = pk.column if (pk := schema.get_primary_key(tgt)) else None
    path.append(
        ContextRelation(
            parent=DataIdentifier(table=tgt, column=tgt_pk),
            child=DataIdentifier(),
        )
    )
    return nodes, path
