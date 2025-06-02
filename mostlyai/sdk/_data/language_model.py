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
import uuid

import pandas as pd

from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.util.common import TABLE_COLUMN_INFIX, TEMPORARY_PRIMARY_KEY
from mostlyai.sdk.domain import ModelType

_LOG = logging.getLogger(__name__)


def split_language_model(
    schema: Schema,
    tgt: str,
    tgt_data: pd.DataFrame,
    ctx_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split LANGUAGE cols into `tgt_data`, and add all other columns to `ctx_data`.

    :return: ctx_data, tgt_data
    """
    enctypes = schema.tables[tgt].encoding_types
    language_cols = [col for col in enctypes if enctypes[col].startswith(ModelType.language)]
    if len(language_cols) == 0:
        # if no LANGUAGE columns are present, then leave data as-is
        return ctx_data, tgt_data
    _LOG.info("split_language_model")
    # split into LANGUAGE and TABULAR columns
    other_data = tgt_data[[c for c in tgt_data if c not in language_cols]]
    # context data must be prefixed
    other_data = other_data.add_prefix(tgt + TABLE_COLUMN_INFIX)
    # No LANGUAGE cols on tgt_data means pull ctx only, can't get LANGUAGE cols from tgt_data, so we must create it
    if set(language_cols).issubset(set(tgt_data.columns)):
        tgt_data = tgt_data[language_cols]
    else:
        tgt_data = pd.DataFrame()

    if ctx_data is None:
        # handle single table case: split all TABULAR columns into `ctx_data`,
        # and only keep txt_data as `tgt_data`.
        ctx_data = other_data
    else:
        # handle two table case: right-join all TABULAR columns to ctx_data,
        # and only keep txt_data as tgt
        ctx_relation = schema.get_parent_context_relation(tgt)

        if ctx_relation:
            ctx_pk = ctx_relation.parent.ref_name()
            tgt_fk = ctx_relation.child.ref_name()
            ctx_data = pd.merge(ctx_data, other_data, how="inner", left_on=ctx_pk, right_on=tgt_fk)

    tmp_keys = [str(uuid.uuid4()) for _ in range(len(ctx_data))]
    tgt_data.insert(0, TEMPORARY_PRIMARY_KEY, tmp_keys)
    ctx_data.insert(0, f"{tgt}{TABLE_COLUMN_INFIX}{TEMPORARY_PRIMARY_KEY}", tmp_keys)
    return ctx_data, tgt_data


def drop_language_columns_in_target(
    tgt: str,
    schema: Schema,
    tgt_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop language columns when pulling data for a tabular model.

    :param tgt: target table
    :param schema: database schema
    :param tgt_data: the target DataFrame
    :return: the target DataFrame with unsupported encoding types dropped
    """
    tgt_table = schema.tables[tgt]
    drop_columns = []
    for col_name, encoding_type in tgt_table.encoding_types.items():
        if encoding_type.startswith(ModelType.language):
            drop_columns.append(col_name)
    if drop_columns:
        _LOG.info(f"drop LANGUAGE columns from target: {drop_columns}")
    return tgt_data.drop(columns=drop_columns)
