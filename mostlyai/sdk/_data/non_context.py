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
from copy import copy

import pandas as pd

from mostlyai.sdk._data.base import DataIdentifier, DataTable, NonContextRelation, Schema

_LOG = logging.getLogger(__name__)

# PULL


def handle_non_context_relations(
    schema: Schema,
    table_name: str,
    data: pd.DataFrame,
    is_target: bool,
) -> pd.DataFrame:
    """Handle all non-context relations for a table"""
    non_context_relations = schema.subset(
        relation_type=NonContextRelation,
        relations_to=[table_name],
    ).relations
    for relation in non_context_relations:
        data = handle_non_context_relation(
            data=data,
            table=schema.tables[relation.parent.table],
            relation=relation,
            is_target=is_target,
        )
    return data


def handle_non_context_relation(
    data: pd.DataFrame,
    table: DataTable,
    relation: NonContextRelation,
    is_target: bool = False,
) -> pd.DataFrame:
    """Handle a single non-context relation for a table and add an is_null column."""
    _LOG.info(f"handle non-context relation {table.name}")

    assert isinstance(relation, NonContextRelation)
    fk = relation.child.ref_name(prefixed=not is_target)
    if fk not in data:
        return data  # nothing to handle

    # identify which values in the FK column have no corresponding entry in the non-context table
    keys = set(data[fk].dropna())
    if len(keys) > 0:
        pk = table.primary_key
        pk_qual_name = DataIdentifier(table.name, pk).ref_name()
        # check for keys not in the parent table
        missing_keys = keys - set(
            table.read_data_prefixed(
                where={pk: list(keys)},
                columns=[pk],
                do_coerce_dtypes=True,
            )[pk_qual_name]
        )
    else:
        missing_keys = set()

    # create the is_null column based on whether a non-context foreign-key is present or not
    is_null_values = data[fk].apply(lambda x: str(pd.isna(x) or x in missing_keys))

    # replace the fk column with the is_null values and rename it accordingly
    data[fk] = is_null_values
    data.rename(columns={fk: relation.get_is_null_column(is_target=is_target)}, inplace=True)

    return data


# POSTPROC


def sample_non_context_keys(
    tgt_is_null: pd.Series,
    non_ctx_pks: pd.DataFrame,
) -> pd.Series:
    """
    Non-context matching algorithm. For each row in tgt_data, we randomly match a record in non_ctx_data.
    Returns pd.Series of sampled row indexes.
    """
    tgt_is_null = tgt_is_null.astype("string")
    # initialize returned pd.Series with NAs
    pk_dtype = non_ctx_pks.convert_dtypes(dtype_backend="pyarrow").dtype
    sampled_keys = pd.Series([pd.NA] * len(tgt_is_null), dtype=pk_dtype, index=tgt_is_null.index)
    # return immediately if no candidates to sample from
    if len(tgt_is_null) == 0:
        return sampled_keys
    tgt_to_sample = tgt_is_null[tgt_is_null != "True"].index
    samples = non_ctx_pks.sample(n=len(tgt_to_sample), replace=True).reset_index(drop=True)
    sampled_keys[tgt_to_sample] = samples
    return sampled_keys


def postproc_non_context(
    tgt_data: pd.DataFrame,
    generated_data_schema: Schema,
    tgt: str,
) -> pd.DataFrame:
    """
    Apply non-context keys allocation for each non-context relation for a generated table.
    """
    tgt_data = copy(tgt_data)
    for rel in generated_data_schema.relations:
        if not isinstance(rel, NonContextRelation) or rel.child.table != tgt:
            continue
        tgt_fk_name = rel.child.column
        tgt_is_null_column_name = rel.get_is_null_column()
        _LOG.info(f"sample non-context keys for {tgt_fk_name}")
        tgt_is_null = tgt_data[tgt_is_null_column_name]
        # read referenced table's keys
        non_ctx_pk_name = rel.parent.column
        non_ctx_pks = generated_data_schema.tables[rel.parent.table].read_data(
            do_coerce_dtypes=True, columns=[non_ctx_pk_name]
        )[non_ctx_pk_name]
        # sample non-ctx keys
        sampled_keys = sample_non_context_keys(tgt_is_null, non_ctx_pks)
        # replace is_null column with sampled keys
        tgt_data.insert(tgt_data.columns.get_loc(tgt_is_null_column_name), tgt_fk_name, sampled_keys)
        tgt_data = tgt_data.drop(columns=[tgt_is_null_column_name])
    return tgt_data
