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
import pytest

from mostlyai.sdk._data.base import ForeignKey, Schema
from mostlyai.sdk._data.context import add_gpc_context, add_ns_context
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.pull_context import pull_context
from mostlyai.sdk.domain import ModelEncodingType


@pytest.fixture
def three_level_schema(tmp_path):
    data = {
        "customers": pd.DataFrame({"id": ["c1", "c2"], "segment": ["A", "B"]}),
        "orders": pd.DataFrame({"id": ["o1", "o2"], "customer_id": ["c1", "c1"], "channel": ["web", "store"]}),
        "items": pd.DataFrame({"id": ["i1"], "order_id": ["o1"], "qty": [1]}),
    }
    tables = {}
    for name, df in data.items():
        path = tmp_path / f"{name}.parquet"
        df.to_parquet(path, index=False)
        parent = {"orders": ("customer_id", "customers"), "items": ("order_id", "orders")}.get(name)
        tables[name] = ParquetDataTable(
            path=path,
            name=name,
            primary_key="id",
            foreign_keys=[ForeignKey(column=parent[0], referenced_table=parent[1], is_context=True)] if parent else [],
            encoding_types={
                col: ModelEncodingType.tabular_categorical for col in df if col not in ["id", "customer_id", "order_id"]
            },
        )
    return Schema(tables=tables)


def test_empty_grandparent_context(three_level_schema):
    schema = three_level_schema
    parents = schema.tables["orders"].read_data_prefixed().iloc[:0]
    result = add_gpc_context(chunk=parents, schema=schema, tgt="items")
    expected = add_gpc_context(chunk=schema.tables["orders"].read_data_prefixed(), schema=schema, tgt="items").iloc[:0]
    pd.testing.assert_frame_equal(result, expected)


def test_pull_empty_three_level_context(three_level_schema, tmp_path):
    schema = three_level_schema
    parents = schema.tables["orders"]
    pd.read_parquet(parents.path).iloc[:0].to_parquet(parents.path, index=False)
    workspace = tmp_path / "workspace"
    pull_context(tgt="items", schema=schema, workspace_dir=workspace)
    context = pd.read_parquet(workspace / "OriginalData" / "ctx-data")
    assert context.empty
    assert {"orders::id", "customers::segment", "orders$prev::channel"}.issubset(context.columns)


@pytest.mark.parametrize("empty", [True, False])
def test_nested_sequence_context(three_level_schema, empty):
    schema = three_level_schema
    parents = schema.tables["orders"].read_data_prefixed()
    context = add_gpc_context(chunk=parents, schema=schema, tgt="items")
    if empty:
        context = context.iloc[:0]
    result = add_ns_context(tgt="items", schema=schema, ctx_data=context)
    assert "orders$prev::channel" in result
    assert len(result) == (0 if empty else 2)
    if not empty:
        assert result["orders::channel"].tolist() == ["web", "store"]
