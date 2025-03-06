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

import pandas as pd

from mostlyai.sdk._data.base import DataIdentifier, Schema, NonContextRelation, ForeignKey
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.non_context import (
    handle_non_context_relation,
    handle_non_context_relations,
    postproc_non_context,
    sample_non_context_keys,
)


def test_handle_non_context_relation(tmp_path):
    """Test single non-context relation with missing and broken links"""
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "non_ctx_id": [1, 1, 2, pd.NA, -1],
            "int": [10, 20, 30, 40, 50],
        }
    ).convert_dtypes()
    non_ctx_df = pd.DataFrame(
        {
            "non_ctx_id": [1, 2, 3, 4],
            "str": ["x1", pd.NA, "x3", "x4"],
            "int": [10, 20, 30, 40],
        }
    ).convert_dtypes()
    non_ctx_df.to_parquet(tmp_path / "non_ctx.parquet")
    non_context_table = ParquetDataTable(path=tmp_path / "non_ctx.parquet", primary_key="non_ctx_id")
    relation = NonContextRelation(
        parent=DataIdentifier(table="non_ctx", column="int"),
        child=DataIdentifier(table="tgt", column="non_ctx_id"),
    )
    enriched_data = handle_non_context_relation(
        data=data,
        table=non_context_table,
        relation=relation,
        is_target=True,
    )
    data_expected = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "non_ctx_id.non_ctx._is_null": ["False", "False", "False", "True", "True"],
            "int": [10, 20, 30, 40, 50],
        }
    ).convert_dtypes()
    pd.testing.assert_frame_equal(enriched_data, data_expected, check_dtype=False)


def test_handle_non_context_relations(tmp_path):
    """Test multiple non-context relations"""
    # prepare data
    non_ctx_path = tmp_path / "non_ctx.parquet"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "int": [42, 42, 42, 42, 42],
        }
    ).to_parquet(non_ctx_path)

    tgt_path = tmp_path / "tgt.parquet"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "non_ctx1_id": [1, 2, 3, 4, 5],  # non-context
            "non_ctx2_id": [1, 2, 3, 4, 5],  # non-context
            "tgt_id": [1, 2, 3, 4, 5],  # self-referential
            "int": [42, 42, 42, 42, 42],
        }
    ).to_parquet(tgt_path)

    tables = {
        "non_ctx": ParquetDataTable(path=non_ctx_path, primary_key="id", name="non_ctx"),
        "tgt": ParquetDataTable(
            path=tgt_path,
            primary_key="id",
            name="tgt",
            foreign_keys=[
                ForeignKey(column="non_ctx1_id", referenced_table="non_ctx", is_context=False),
                ForeignKey(column="non_ctx2_id", referenced_table="non_ctx", is_context=False),
                ForeignKey(column="tgt_id", referenced_table="tgt", is_context=False),
            ],
        ),
    }

    schema = Schema(tables=tables)

    data = schema.tables["tgt"].read_data_prefixed(include_table_prefix=False)
    data = handle_non_context_relations(
        schema=schema,
        table_name="tgt",
        data=data,
        is_target=True,
    )
    assert list(data.columns) == [
        "id",
        "non_ctx1_id.non_ctx._is_null",
        "non_ctx2_id.non_ctx._is_null",
        "tgt_id.tgt._is_null",
        "int",
    ]


def test_sample_non_context_keys(tmp_path):
    tgt_is_null = pd.Series(["True", "False", "False", "False", "False"])
    non_ctx_pks = pd.Series(["r0", "r1", "r2", "r3"])
    sampled_keys = sample_non_context_keys(tgt_is_null, non_ctx_pks)
    assert sampled_keys.isna().to_list() == [True, False, False, False, False]
    assert all(sampled_keys[1:].isin(non_ctx_pks))


def test_sample_non_context_keys_all_null(tmp_path):
    tgt_is_null = pd.Series(["True", "True"])
    non_ctx_pks = pd.Series(["r0", "r1"])
    sampled_keys = sample_non_context_keys(tgt_is_null, non_ctx_pks)
    assert sampled_keys.isna().to_list() == [True, True]


def test_postproc_non_context(tmp_path):
    # create data with multiple non-context relations
    tgt_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "father.non_ctx._is_null": ["False", "False", "False"],
            "mother.non_ctx._is_null": ["False", "False", "False"],
            "uncle.non_ctx._is_null": ["True", "False", "False"],
        }
    )
    tgt_data.to_parquet(tmp_path / "tgt.parquet")
    non_ctx_data = pd.DataFrame(
        {
            "id": ["c0", "c1", "c2"],
        }
    )
    non_ctx_data.to_parquet(tmp_path / "non_ctx.parquet")
    schema = Schema(
        tables={
            "tgt": ParquetDataTable(
                path=tmp_path / "tgt.parquet",
                name="tgt",
                foreign_keys=[
                    ForeignKey(column="father", referenced_table="non_ctx", is_context=False),
                    ForeignKey(column="mother", referenced_table="non_ctx", is_context=False),
                    ForeignKey(column="uncle", referenced_table="non_ctx", is_context=False),
                ],
            ),
            "non_ctx": ParquetDataTable(path=tmp_path / "non_ctx.parquet", name="non_ctx", primary_key="id"),
        }
    )

    # sample non-context keys
    tgt_postprocessed_data = postproc_non_context(
        tgt_data=tgt_data,
        generated_data_schema=schema,
        tgt="tgt",
    )
    # check post-processed data
    assert tgt_postprocessed_data["uncle"].isna()[0]
    assert not tgt_postprocessed_data["uncle"].isna()[1]
    assert "uncle._is_null" not in tgt_postprocessed_data.columns
