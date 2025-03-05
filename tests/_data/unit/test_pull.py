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

import json
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mostlyai.sdk._data import pull, pull_context
from mostlyai.sdk.domain import ModelEncodingType
from mostlyai.sdk._data.base import (
    Schema,
    ForeignKey,
)
from mostlyai.sdk._data.util.common import TEMPORARY_PRIMARY_KEY
from mostlyai.sdk._data.dtype import (
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
    is_timestamp_dtype,
    STRING,
)
from mostlyai.sdk._data.file.table.csv import CsvDataTable
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.pull_utils import (
    MAX_SAMPLES_PER_ROOT,
    determine_n_partitions,
    mask_keys,
)
from pandas.testing import assert_series_equal


PULL_MODULE = "mostlyai.sdk._data.pull_utils"


def read_json(path: Path, default: dict | None = None, raises: bool | None = None) -> dict:
    """
    Reads JSON.

    :param path: path to json
    :param default: default used in case path does not exist
    :param raises: if True, raises exception if path does not exist,
        otherwise returns default
    :return: dict representation of JSON
    """

    if default is None:
        default = {}
    if not path.exists():
        if raises:
            raise RuntimeError(f"File [{path}] does not exist")
        else:
            return default
    with open(path) as json_file:
        data = json.load(json_file)
    return data


class DisableMaskKeys:
    @pytest.fixture(autouse=True)
    def setup_mask_keys(self):
        with mock.patch(f"{PULL_MODULE}.mask_keys") as mock_mask_keys:
            # Configure the mock to return the input arguments
            def mock_mask_keys_function(ctx_data=None, tgt_data=None, **_):
                return ctx_data, tgt_data

            mock_mask_keys.side_effect = mock_mask_keys_function
            yield


def all_values_uuid_length(df):
    uuid_length = 36  # Typical length of a UUID string
    for column in df.columns:
        if not df[column].apply(lambda x: len(str(x)) == uuid_length).all():
            return False
    return True


class TestPartitioning:
    @pytest.fixture
    def gpc_ctx_tgt_schema(self):
        tables = {
            "gpc": CsvDataTable(
                path="gpc.csv",
                name="gpc",
                primary_key="id",
                columns=["id", "str", "dt"],
                foreign_keys=[],
            ),
            "ctx": CsvDataTable(
                path="ctx.csv",
                name="ctx",
                primary_key="id",
                columns=["id", "gpc_id", "num"],
                foreign_keys=[
                    ForeignKey(column="gpc_id", referenced_table="gpc", is_context=True),
                ],
            ),
            "tgt": CsvDataTable(
                path="tgt.csv",
                name="tgt",
                columns=["ctx_id", "na"],
                foreign_keys=[
                    ForeignKey(column="ctx_id", referenced_table="ctx", is_context=True),
                ],
            ),
        }
        schema = Schema(tables=tables)
        return schema

    @pytest.mark.parametrize(
        "ctx_n_rows,tgt_n_rows,exp_n_partitions",
        [(0, 0, 1), (1, 1, 1), (300_000, 3_000_000, 3), (1_000_000, 49_000_000, 32)],
    )
    def test_determine_n_partitions(self, gpc_ctx_tgt_schema, ctx_n_rows, tgt_n_rows, exp_n_partitions):
        ctx_tgt_nodes = ["gpc", "ctx", "tgt"]
        n_partitions = determine_n_partitions(
            schema=gpc_ctx_tgt_schema,
            ctx_nodes=ctx_tgt_nodes[:-1],
            tgt_node=ctx_tgt_nodes[-1],
            ctx_n_rows=ctx_n_rows,
            tgt_n_rows=tgt_n_rows,
        )
        # bound to the specific constants embedded in determine_n_partitions
        assert n_partitions == exp_n_partitions


class TestPullSingle(DisableMaskKeys):
    def create_single_table_schema(self, path, tgt_df, tgt_pk=None):
        tgt_path = Path(path) / "tgt.parquet"
        tgt_df.to_csv(tgt_path, index=False)
        tables = {
            "tgt": CsvDataTable(path=tgt_path, primary_key=tgt_pk, name="tgt"),
        }
        schema = Schema(tables=tables)
        return schema

    @pytest.fixture
    def single_table_data(self):
        tgt_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "bool": [True, False, True, False, True, False],
                "int": [10, 20, 30, 40, 50, 60],
                "float": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "str": ["cat", "dog", "cat", "dog", "cat", "dog"],
                "curr": ["$12", "$23.3", "$5", "-$5", "", pd.NA],
                "date": [
                    "2024-01-01",
                    "2024-05-24",
                    "2025-05-24",
                    "2026-12-24",
                    pd.NaT,
                    pd.NaT,
                ],
                "date_fmt": [
                    "01 Jan 2024",
                    "01 Mar 2024",
                    "31 May 2024",
                    "12 Aug 2024",
                    pd.NaT,
                    pd.NaT,
                ],
                "ts": [
                    "2024-01-01 08:30:32",
                    "2024-05-24 08:30:32",
                    "2025-05-24 08:30:32",
                    "2026-12-24 08:30:32",
                    pd.NaT,
                    pd.NaT,
                ],
                "bigint": [1372636858620000589, -3, -2, -1, pd.NA, pd.NA],
                "a.dot.col": ["a", "b", "c", "d", "e", "f"],
            }
        )
        return tgt_df

    def test_pull_simple(self, tmp_path, single_table_data):
        tgt_df = single_table_data
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk="id")
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        tgt_data_path = tmp_path / "OriginalData" / "tgt-data"
        tgt_data = pd.read_parquet(tgt_data_path)
        assert len(tgt_df) == len(tgt_data)
        assert is_string_dtype(tgt_data["id"])
        assert is_string_dtype(tgt_data["bool"])
        assert is_integer_dtype(tgt_data["int"])
        assert is_float_dtype(tgt_data["float"])
        assert is_string_dtype(tgt_data["str"])
        assert is_timestamp_dtype(tgt_data["date"])
        assert is_timestamp_dtype(tgt_data["ts"])
        assert is_integer_dtype(tgt_data["bigint"])
        assert is_string_dtype(tgt_data["a.dot.col"])
        # regression test of pyarrow[int64] issues; see https://github.com/apache/arrow/issues/35273
        assert tgt_data["bigint"].round() is not None
        # check pulled meta data
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        assert tgt_keys == {"primary_key": "id"}
        enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert enctypes["bool"] == ModelEncodingType.tabular_categorical
        assert enctypes["int"] == ModelEncodingType.tabular_numeric_auto
        assert enctypes["float"] == ModelEncodingType.tabular_numeric_auto
        assert enctypes["str"] == ModelEncodingType.tabular_categorical
        assert enctypes["date"] == ModelEncodingType.tabular_datetime
        assert enctypes["bigint"] == ModelEncodingType.tabular_numeric_auto
        assert enctypes["a.dot.col"] == ModelEncodingType.tabular_categorical
        assert "id" not in enctypes

    def test_pull_columns(self, tmp_path, single_table_data):
        tgt_df = single_table_data
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk="id")
        # pull data
        columns = ["str", "int"]
        schema.tables["tgt"].columns = columns
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert set(tgt_data.columns) == {f"{c}" for c in columns}
        # check pulled meta data
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        assert len(tgt_keys) == 0  # primary key was not pulled
        enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert len(enctypes) == len(columns)

    def test_pull_encoding_types(self, tmp_path, single_table_data):
        tgt_df = single_table_data
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk="id")
        # pull data
        encoding_types = {
            "int": ModelEncodingType.tabular_categorical,
            "str": ModelEncodingType.auto,
            "date_fmt": ModelEncodingType.tabular_datetime_relative,  # unsupported for subject tables
        }
        schema.tables["tgt"].columns = list(encoding_types.keys())
        schema.tables["tgt"].encoding_types = encoding_types
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert is_string_dtype(tgt_data["int"])
        assert is_string_dtype(tgt_data["str"])
        assert is_timestamp_dtype(tgt_data["date_fmt"])
        # check pulled meta data
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert tgt_enctypes["int"] == ModelEncodingType.tabular_categorical
        assert tgt_enctypes["str"] == ModelEncodingType.tabular_categorical
        assert tgt_enctypes["date_fmt"] == ModelEncodingType.tabular_datetime  # fallback check

    @pytest.mark.parametrize("max_sample_size", [0, 3, 5, 1000])
    def test_pull_max_sample_size(self, tmp_path, single_table_data, max_sample_size):
        tgt_df = single_table_data
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk="id")
        # pull data
        pull(
            tgt="tgt",
            schema=schema,
            max_sample_size=max_sample_size,
            workspace_dir=tmp_path,
        )
        # check pulled data
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        expected_sample_size = min(max(1, max_sample_size), len(tgt_df))
        assert len(tgt_data) == expected_sample_size

    @pytest.mark.parametrize("tgt_pk", [None, "id"])
    def test_pull_is_shuffled(self, tmp_path, single_table_data, tgt_pk):
        tgt_df = pd.DataFrame({"id": list(range(1000))})
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk=tgt_pk)
        # pull data without sample_size
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        # assert non-duplicates keys
        assert len(set(tgt_data["id"])) == len(tgt_df)
        # assert random order
        assert tgt_data["id"].head(5).astype("int").max() > 5
        # pull data with sample_size
        max_sample_size = 5
        pull(
            tgt="tgt",
            schema=schema,
            max_sample_size=max_sample_size,
            workspace_dir=tmp_path,
        )
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        # assert non-duplicate keys
        assert len(set(tgt_data["id"])) == max_sample_size
        # assert random order
        assert tgt_data["id"].astype("int").max() > max_sample_size

    @pytest.mark.parametrize("no_rows", [0, 1])
    @pytest.mark.parametrize("pk_col", [None, "a"])
    def test_pull_emptyish_tgt(self, tmp_path, no_rows, pk_col):
        # create empty-ish DataFrame
        def vals(col):
            return [f"{col}{idx}" for idx in range(no_rows)]

        tgt_df = pd.DataFrame({"a": vals("a"), "b": vals("b")})
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk=pk_col)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert tgt_data.shape == tgt_df.shape
        # check pulled meta data
        encoding_types = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert len(encoding_types) == (2 if pk_col is None else 1)

    def test_pull_duplicate_keys(self, tmp_path):
        tgt_df = pd.DataFrame({"id": [1, 1, 2]})
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk="id")
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check that we didn't drop the duplicate key in tgt
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(tgt_data) == 3

    @pytest.mark.parametrize("tgt_pk", [None, "id"])
    def test_pull_partitioned(self, tmp_path, single_table_data, tgt_pk):
        tgt_df = single_table_data
        schema = self.create_single_table_schema(tmp_path, tgt_df, tgt_pk=tgt_pk)
        n_partitions = 3
        with patch(f"{PULL_MODULE}.determine_n_partitions", return_value=n_partitions):
            # pull data
            pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        tgt_data_path = tmp_path / "OriginalData" / "tgt-data"
        tgt_data = pd.read_parquet(tgt_data_path)
        assert len(tgt_data) == len(tgt_df)


class TestPullSequential(DisableMaskKeys):
    def create_two_table_schema(self, path, ctx_df, tgt_df, ctx_pk="id", tgt_fk="ctx_id", tgt_pk=None):
        # Define paths for context and target tables
        ctx_path = Path(path) / "ctx.parquet"
        tgt_path = Path(path) / "tgt.parquet"

        # Save data to respective parquet files
        ctx_df.to_parquet(ctx_path)
        tgt_df.to_parquet(tgt_path)

        # Create context table
        ctx_table = ParquetDataTable(path=ctx_path, primary_key=ctx_pk, name="ctx")

        # Create target table with foreign key referencing the context table
        tgt_table = ParquetDataTable(
            path=tgt_path,
            primary_key=tgt_pk,
            name="tgt",
            foreign_keys=[ForeignKey(column=tgt_fk, referenced_table="ctx", is_context=True)],
        )

        # Create schema with the two tables
        tables = {"ctx": ctx_table, "tgt": tgt_table}
        schema = Schema(tables=tables)

        return schema

    @pytest.fixture
    def two_table_data(self):
        # 10 context records
        # 5x with seq_len=10, 5x with seq_len=0
        ctx_df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "int": [42] * 10,
                "str": ["a"] * 10,
            }
        )
        tgt_df = pd.DataFrame(
            {
                "ctx_id": [0, 1, 2, 3, 4] * 10,
                "id": list(range(50)),
                "int": [10, 11, 12, 13, 14] * 10,
                "str": ["a", "b", "c", "d", "e"] * 10,
            }
        )
        return ctx_df, tgt_df

    def test_pull_simple(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df, tgt_pk="id")
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert ctx_data.shape == ctx_df.shape
        assert tgt_data.shape == tgt_df.shape
        # assert that all context keys are retained
        assert all(ctx_data["ctx::id"].isin(ctx_df["id"].astype("string")))
        # assert that only target keys are retained, that also exist in ctx
        assert all(tgt_data["ctx_id"].isin(ctx_df["id"].astype("string")))
        # assert dtypes
        assert is_string_dtype(tgt_data["id"])
        assert is_string_dtype(tgt_data["ctx_id"])
        assert is_integer_dtype(ctx_data["ctx::int"])
        assert is_string_dtype(ctx_data["ctx::str"])
        assert is_integer_dtype(tgt_data["int"])
        assert is_string_dtype(tgt_data["str"])
        # check pulled meta data - keys.json
        ctx_keys = read_json(tmp_path / "OriginalData" / "ctx-meta" / "keys.json")
        assert ctx_keys == {"primary_key": "ctx::id", "root_key": "ctx::id"}
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        assert tgt_keys == {"primary_key": "id", "context_key": "ctx_id"}
        # check pulled meta data - encoding-types.json
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        assert "ctx::id" not in ctx_enctypes
        assert ctx_enctypes["ctx::int"] == ModelEncodingType.tabular_numeric_auto
        assert ctx_enctypes["ctx::str"] == ModelEncodingType.tabular_categorical
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert "ctx_id" not in tgt_enctypes
        assert "id" not in tgt_enctypes
        assert tgt_enctypes["int"] == ModelEncodingType.tabular_numeric_auto
        assert tgt_enctypes["str"] == ModelEncodingType.tabular_categorical

    def test_pull_columns(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        schema.tables["ctx"].columns = ["id", "int"]
        schema.tables["tgt"].columns = ["ctx_id", "int"]
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        # assert that only max_sample_size and subset of columns is pulled
        assert set(ctx_data.columns) == {"ctx::id", "ctx::int"}
        assert set(tgt_data.columns) == {"ctx_id", "int"}
        # check pulled meta data
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        assert "ctx::str" not in ctx_enctypes
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert "str" not in tgt_enctypes

    def test_pull_encoding_types(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        schema.tables["ctx"].encoding_types |= {
            "int": ModelEncodingType.tabular_categorical,
            "str": ModelEncodingType.tabular_numeric_auto,
        }
        schema.tables["tgt"].encoding_types |= {
            "int": ModelEncodingType.tabular_categorical,
            "str": ModelEncodingType.tabular_numeric_auto,
        }
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert is_string_dtype(ctx_data["ctx::int"])
        assert is_float_dtype(ctx_data["ctx::str"])
        assert is_string_dtype(tgt_data["int"])
        assert is_float_dtype(tgt_data["str"])
        # check pulled meta data
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        assert ctx_enctypes["ctx::int"] == ModelEncodingType.tabular_categorical
        assert ctx_enctypes["ctx::str"] == ModelEncodingType.tabular_numeric_auto
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert tgt_enctypes["int"] == ModelEncodingType.tabular_categorical
        assert tgt_enctypes["str"] == ModelEncodingType.tabular_numeric_auto

    def test_pull_unsupported_encoding_types_from_context(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        schema.tables["ctx"].encoding_types |= {"int": ModelEncodingType.language_text}
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert "ctx::int" not in ctx_data
        # check pulled meta data
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        assert "ctx::int" not in ctx_enctypes

    @pytest.mark.parametrize("max_sample_size", [0, 3, 5, 1000])
    def test_pull_max_sample_size(self, tmp_path, two_table_data, max_sample_size):
        ctx_df, tgt_df = two_table_data
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        pull(
            tgt="tgt",
            schema=schema,
            max_sample_size=max_sample_size,
            workspace_dir=tmp_path,
        )
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        # assert that only max_sample_size is pulled
        expected_sample_size = min(max(1, max_sample_size), len(ctx_df))
        assert len(ctx_data) == expected_sample_size
        assert len(set(tgt_data["ctx_id"])) <= max(1, max_sample_size)
        # assert that keys in context are unique
        assert len(set(ctx_data["ctx::id"])) == expected_sample_size
        # assert that only target keys are retained that also exist in ctx
        assert all(tgt_data["ctx_id"].isin(ctx_data["ctx::id"]))

    @pytest.mark.parametrize("max_sample_size", [None, 6])
    def test_pull_is_shuffled(self, tmp_path, two_table_data, max_sample_size):
        ctx_df = pd.DataFrame({"id": list(range(1000))})
        tgt_df = pd.DataFrame({"ctx_id": list(range(1000))})
        tgt_df["idx"] = tgt_df.groupby("ctx_id").cumcount()
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        pull(
            tgt="tgt",
            schema=schema,
            max_sample_size=max_sample_size,
            workspace_dir=tmp_path,
        )
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        # assert non-duplicate context keys
        assert len(set(ctx_data["ctx::id"])) == (max_sample_size or len(ctx_df))
        assert len(set(tgt_data["ctx_id"])) <= (max_sample_size or len(ctx_df))
        # assert random order of context
        assert ctx_data["ctx::id"].head(5).astype("int").max() > 5
        # assert that order of tgt records is retained
        all(tgt_data.groupby("ctx_id")["idx"].diff().dropna() == 1)

    def test_pull_empty_tgt(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        tgt_df = tgt_df.drop(tgt_df.index)  # drop records from tgt_df
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert ctx_data.shape == ctx_df.shape
        assert tgt_data.shape == (0, tgt_df.shape[1])

    def test_pull_empty_ctx(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        ctx_df = ctx_df.drop(ctx_df.index)  # drop records from ctx_df
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert ctx_data.shape == (0, ctx_df.shape[1])
        assert tgt_data.shape == (0, tgt_df.shape[1])

    def test_pull_key_dtype_mismatch(self, tmp_path, two_table_data):
        # test scenario where PK is a string, and FK is an integer
        ctx_df, tgt_df = two_table_data
        ctx_df["id"] = ctx_df["id"].astype("string")
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert ctx_data.shape == ctx_df.shape
        assert tgt_data.shape == tgt_df.shape

    def test_pull_drop_orphans(self, tmp_path, two_table_data):
        # test scenario where tgt contains records without a key in context
        ctx_df, tgt_df = two_table_data
        extra_row = tgt_df.iloc[0:1]
        extra_row["id"] = -1
        tgt_df = pd.concat([tgt_df, extra_row], axis=0)
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert ctx_data.shape == ctx_df.shape
        assert len(tgt_data) == len(tgt_df.loc[tgt_df["ctx_id"].isin(ctx_df["id"])])
        # assert that only target keys are retained, that also exist in ctx
        assert all(tgt_data["ctx_id"].isin(ctx_df["id"].astype("string")))

    def test_pull_no_tgt_primary_key(self, tmp_path, two_table_data):
        ctx_df, tgt_df = two_table_data
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df, tgt_pk=None)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert ctx_data.shape == ctx_df.shape
        assert tgt_data.shape == tgt_df.shape

    def test_pull_duplicate_keys(self, tmp_path):
        ctx_df = pd.DataFrame({"id": [1, 1, 2]})
        tgt_df = pd.DataFrame({"ctx_id": [1, 2, 2], "id": [1, 1, 2]})
        schema = self.create_two_table_schema(tmp_path, ctx_df, tgt_df, tgt_pk="id")
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check that we silently dropped the duplicate key
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_data) == 2
        # check that we didn't drop the duplicate key in tgt
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(tgt_data) == 3

    def test_max_samples_per_root_language_model(self, tmp_path):
        gpc_df = pd.DataFrame({"id": [0, 1]})
        ctx_df = pd.DataFrame({"gpc_id": [0, 1] * 3 + [0] * 4, "id": range(10)})

        # tgt_df with increasing repetitions of ctx_id from 1 to 10
        tgt_data = []
        for ctx_id in ctx_df["id"]:
            repetitions = min(ctx_id % 10 + 1, 10)  # from 1 to 10 based on ctx_id
            for _ in range(repetitions):
                tgt_data.append({"ctx_id": ctx_id, "col": f"{len(tgt_data)}"})

        tgt_df = pd.DataFrame(tgt_data)

        schema = create_three_table_schema(tmp_path, gpc_df, ctx_df, tgt_df)
        schema.tables["tgt"].encoding_types["col"] = ModelEncodingType.language_text
        pull(tgt="tgt", schema=schema, model_type="LANGUAGE", workspace_dir=tmp_path)
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        data = pd.merge(
            ctx_data,
            tgt_data,
            left_on=f"tgt::{TEMPORARY_PRIMARY_KEY}",
            right_on=TEMPORARY_PRIMARY_KEY,
        )
        data_samples_per_root = data.groupby("gpc::id").size().reset_index(name="samples")
        # allow some slack on max samples per root due to probabilistic capping for tgt being a language model
        assert (data_samples_per_root["samples"] <= (MAX_SAMPLES_PER_ROOT + 2)).all()


def create_three_table_schema(
    path,
    gpc_df,
    ctx_df,
    tgt_df,
    gpc_name="gpc",
    ctx_name="ctx",
    tgt_name="tgt",
    gpc_pk="id",
    ctx_fk="gpc_id",
    ctx_pk="id",
    tgt_fk="ctx_id",
    tgt_pk=None,
):
    gpc_path = Path(path) / f"{gpc_name}.parquet"
    ctx_path = Path(path) / f"{ctx_name}.parquet"
    tgt_path = Path(path) / f"{tgt_name}.parquet"

    gpc_df.to_parquet(gpc_path)
    ctx_df.to_parquet(ctx_path)
    tgt_df.to_parquet(tgt_path)

    gpc_table = ParquetDataTable(path=gpc_path, primary_key=gpc_pk, name=gpc_name)

    ctx_table = ParquetDataTable(
        path=ctx_path,
        primary_key=ctx_pk,
        name=ctx_name,
        foreign_keys=[ForeignKey(column=ctx_fk, referenced_table=gpc_name, is_context=True)],
    )

    tgt_table = ParquetDataTable(
        path=tgt_path,
        primary_key=tgt_pk,
        name=tgt_name,
        foreign_keys=[ForeignKey(column=tgt_fk, referenced_table=ctx_name, is_context=True)],
    )

    tables = {gpc_name: gpc_table, ctx_name: ctx_table, tgt_name: tgt_table}

    schema = Schema(tables=tables)
    return schema


class TestPullThreeTableHierarchy(DisableMaskKeys):
    @pytest.fixture
    def three_table_data(self):
        # 10 grandparent context records
        # each with 10 context records
        # half of those with 10 target records
        gpc_df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "int": [42] * 10,
                "str": ["a"] * 10,
            }
        )
        ctx_df = pd.DataFrame(
            {
                "gpc_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,
                "id": list(range(100)),
                "int": [42] * 100,
                "str": ["a"] * 100,
            }
        )
        tgt_df = pd.DataFrame(
            {
                "ctx_id": list(range(50)) * 10,
                "id": list(range(500)),
                "int": [42] * 500,
                "str": ["a"] * 500,
            }
        )
        return gpc_df, ctx_df, tgt_df

    def test_pull_simple(self, tmp_path, three_table_data):
        gpc_df, ctx_df, tgt_df = three_table_data
        schema = create_three_table_schema(tmp_path, gpc_df, ctx_df, tgt_df)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(ctx_data) == len(gpc_df) * MAX_SAMPLES_PER_ROOT
        # ensure that we shuffle, i.e. not just sample first ctx_keys for each gpc
        assert ctx_data["ctx::id"].astype(int).max() > len(gpc_df) * MAX_SAMPLES_PER_ROOT
        # check that no grandparent keys are present
        assert "gpc::id" not in ctx_df
        assert "ctx::gpc_id" not in ctx_df
        # check that MAX_SAMPLES_PER_ROOT is considered
        ctx_df["id"] = ctx_df["id"].astype("string")
        ctx_gpc = ctx_data.merge(ctx_df, left_on="ctx::gpc_id", right_on="id")[["gpc_id"]]
        assert all(ctx_gpc.groupby("gpc_id").size() <= MAX_SAMPLES_PER_ROOT)
        # assert that all context keys are retained
        assert all(ctx_data["ctx::id"].isin(ctx_df["id"].astype("string")))
        # assert dtypes
        assert is_string_dtype(ctx_data["ctx::id"])
        assert is_integer_dtype(ctx_data["ctx::int"])
        assert is_string_dtype(ctx_data["ctx::str"])
        assert is_integer_dtype(ctx_data["gpc::int"])
        assert is_string_dtype(ctx_data["gpc::str"])
        assert is_string_dtype(tgt_data["ctx_id"])
        assert is_integer_dtype(tgt_data["int"])
        assert is_string_dtype(tgt_data["str"])
        # check pulled meta data - keys.json
        ctx_keys = read_json(tmp_path / "OriginalData" / "ctx-meta" / "keys.json")
        assert ctx_keys == {"primary_key": "ctx::id", "root_key": "gpc::id"}
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        assert tgt_keys == {"context_key": "ctx_id"}
        # check pulled meta data - encoding-types.json
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        assert "ctx::id" not in ctx_enctypes
        assert ctx_enctypes["ctx::int"] == ModelEncodingType.tabular_numeric_auto
        assert ctx_enctypes["ctx::str"] == ModelEncodingType.tabular_categorical
        assert ctx_enctypes["gpc::int"] == ModelEncodingType.tabular_numeric_auto
        assert ctx_enctypes["gpc::str"] == ModelEncodingType.tabular_categorical
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert "ctx_id" not in tgt_enctypes
        assert tgt_enctypes["int"] == ModelEncodingType.tabular_numeric_auto
        assert tgt_enctypes["str"] == ModelEncodingType.tabular_categorical

    def test_pull_ctx_only(self, tmp_path, three_table_data):
        gpc_df, ctx_df, tgt_df = three_table_data
        schema = create_three_table_schema(tmp_path, gpc_df, ctx_df, tgt_df)
        # pull data
        pull_context(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        assert not (tmp_path / "OriginalData" / "tgt-data").exists()
        assert not (tmp_path / "OriginalData" / "tgt-meta").exists()
        assert not (tmp_path / "OriginalData" / "ctx-meta").exists()
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_data) == len(ctx_df)
        # check that no grandparent keys are present
        assert "gpc::id" not in ctx_df
        assert "ctx::gpc_id" not in ctx_df
        # assert that all context keys are retained
        assert all(ctx_data["ctx::id"].isin(ctx_df["id"].astype("string")))
        # for (syn) ctx we do not coerce dtypes, as they were presumable coerced before

    @pytest.mark.parametrize("do_ctx_only", [True, False])
    @pytest.mark.parametrize("model_type", ["LANGUAGE", "TABULAR"])
    def test_pull_partitioned(self, tmp_path, three_table_data, do_ctx_only, model_type):
        gpc_df, ctx_df, tgt_df = three_table_data
        schema = create_three_table_schema(tmp_path, gpc_df, ctx_df, tgt_df)
        # pull data, ensuring a particular number of partitions
        n_partitions = 3
        pull_fn = pull_context if do_ctx_only else pull
        with patch(f"{PULL_MODULE}.determine_n_partitions", return_value=n_partitions):
            pull_fn(
                tgt="tgt",
                schema=schema,
                model_type=model_type,
                workspace_dir=tmp_path,
            )

        # check partitions are created properly
        tgt_data_path = tmp_path / "OriginalData" / "tgt-data"
        ctx_data_path = tmp_path / "OriginalData" / "ctx-data"
        if do_ctx_only:
            assert 1 <= len(list(ctx_data_path.glob("*-ctx.parquet"))) <= n_partitions
        else:
            assert 1 <= len(list(tgt_data_path.glob("*-trn.parquet"))) <= n_partitions
            assert 0 <= len(list(tgt_data_path.glob("*-val.parquet"))) <= n_partitions
            assert 1 <= len(list(ctx_data_path.glob("*-trn.parquet"))) <= n_partitions
            assert 0 <= len(list(ctx_data_path.glob("*-val.parquet"))) <= n_partitions

        # check pulled data
        ctx_fn_parts = [f.name for f in (tmp_path / "OriginalData" / "ctx-data").iterdir()]

        if not do_ctx_only:  # for the case of pull_tgt_and_ctx
            assert (tmp_path / "OriginalData" / "ctx-meta").exists()
            assert (tmp_path / "OriginalData" / "tgt-meta").exists()
            tgt_fn_parts = [f.name for f in (tmp_path / "OriginalData" / "tgt-data").iterdir()]
            assert sorted(ctx_fn_parts) == sorted(tgt_fn_parts)

            ctx_df_parts = []
            for fn in tgt_fn_parts:
                ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data" / fn)
                tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data" / fn)
                assert all(tgt_data["ctx_id"].isin(ctx_data["ctx::id"]))
                ctx_df_parts.append(ctx_data)
            ctx_data = pd.concat(ctx_df_parts)
            if model_type == "LANGUAGE":
                assert len(ctx_data) <= len(gpc_df) * MAX_SAMPLES_PER_ROOT
            else:
                assert len(ctx_data) == len(gpc_df) * MAX_SAMPLES_PER_ROOT
        else:  # for the case of pull_ctx_only
            ctx_df_parts = []
            for fn in ctx_fn_parts:
                ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data" / fn)
                ctx_df_parts.append(ctx_data)
            ctx_data = pd.concat(ctx_df_parts)
            assert len(ctx_data) == len(ctx_df)
            assert all(ctx_df["id"].astype(STRING).isin(ctx_data["ctx::id"]))


class TestPullFourTableHierarchy:
    def create_four_table_schema(self, path):
        # Define file paths
        ctx2_path = Path(path) / "ctx2.parquet"
        ctx1_path = Path(path) / "ctx1.parquet"
        ctx0_path = Path(path) / "ctx0.parquet"
        tgt_path = Path(path) / "tgt.parquet"

        # Create parquet files
        pd.DataFrame({"id": [1, 2]}).to_parquet(ctx2_path)
        pd.DataFrame({"ctx2_id": [1, 1], "id": [3, 4]}).to_parquet(ctx1_path)
        pd.DataFrame({"ctx1_id": [3, 3], "id": [5, 6]}).to_parquet(ctx0_path)
        pd.DataFrame({"ctx0_id": [5, 5], "id": [7, 8]}).to_parquet(tgt_path)

        # Define tables with foreign keys
        tables = {
            "ctx2": ParquetDataTable(path=ctx2_path, primary_key="id", name="ctx2", foreign_keys=[]),
            "ctx1": ParquetDataTable(
                path=ctx1_path,
                primary_key="id",
                name="ctx1",
                foreign_keys=[ForeignKey(column="ctx2_id", referenced_table="ctx2", is_context=True)],
            ),
            "ctx0": ParquetDataTable(
                path=ctx0_path,
                primary_key="id",
                name="ctx0",
                foreign_keys=[ForeignKey(column="ctx1_id", referenced_table="ctx1", is_context=True)],
            ),
            "tgt": ParquetDataTable(
                path=tgt_path,
                primary_key="id",
                name="tgt",
                foreign_keys=[ForeignKey(column="ctx0_id", referenced_table="ctx0", is_context=True)],
            ),
        }

        return Schema(tables=tables)

    def test_pull_simple(self, tmp_path):
        schema = self.create_four_table_schema(tmp_path)
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_df = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(ctx_df.columns) == 5  # accumulated ctx columns
        assert len(tgt_df) == 2
        # ensure all keys (which in this case all columns) were masked as UUIDs
        assert all_values_uuid_length(ctx_df)
        assert all_values_uuid_length(tgt_df)


class TestPullNonContext:
    @pytest.fixture
    def multi_parent_schema(self, tmp_path):
        # Create data frames
        non_ctx_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "int": [10, 20, 30],
                "str": ["s1", "s2", "s3"],
            }
        )
        ctx_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "int": [10, 20, 30],
                "str": ["c1", "c2", "c3"],
                "non_ctx_1_id": [1, 1, 2],
                "non_ctx_2_id": [2, 3, 3],
            }
        )
        tgt_df = pd.DataFrame(
            {
                "ctx_id": [1, 1, 2, 3],
                "id": [1, 2, 3, 4],
                "non_ctx_1_id": [1, 1, 2, -1],
                "non_ctx_2_id": [2, 3, 2, pd.NA],
            }
        )

        non_ctx_path = Path(tmp_path) / "non_ctx.parquet"
        ctx_path = Path(tmp_path) / "ctx.parquet"
        tgt_path = Path(tmp_path) / "tgt.parquet"

        non_ctx_df.to_parquet(non_ctx_path)
        ctx_df.to_parquet(ctx_path)
        tgt_df.to_parquet(tgt_path)

        tables = {
            "non_ctx": ParquetDataTable(path=non_ctx_path, primary_key="id", name="non_ctx", foreign_keys=[]),
            "ctx": ParquetDataTable(
                path=ctx_path,
                primary_key="id",
                name="ctx",
                foreign_keys=[
                    ForeignKey(
                        column="non_ctx_1_id",
                        referenced_table="non_ctx",
                        is_context=False,
                    ),
                    ForeignKey(
                        column="non_ctx_2_id",
                        referenced_table="non_ctx",
                        is_context=False,
                    ),
                ],
            ),
            "tgt": ParquetDataTable(
                path=tgt_path,
                primary_key="id",
                name="tgt",
                foreign_keys=[
                    ForeignKey(column="ctx_id", referenced_table="ctx", is_context=True),
                    ForeignKey(
                        column="non_ctx_1_id",
                        referenced_table="non_ctx",
                        is_context=False,
                    ),
                    ForeignKey(column="non_ctx_2_id", referenced_table="ctx", is_context=False),
                ],
            ),
        }

        return Schema(tables=tables)

    @pytest.mark.parametrize("model_type", ["LANGUAGE", "TABULAR"])
    def test_multi_parent(self, tmp_path, multi_parent_schema, model_type):
        schema = multi_parent_schema
        # pull data
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path, model_type=model_type)
        # check pulled data
        pd.read_parquet(schema.tables["non_ctx"].path)
        ctx_df = pd.read_parquet(schema.tables["ctx"].path)
        tgt_df = pd.read_parquet(schema.tables["tgt"].path)
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(ctx_data) == len(ctx_df)
        assert len(tgt_data) == len(tgt_df)
        if model_type == "LANGUAGE":
            assert ctx_data.columns.tolist() == [
                "ctx::id",
                "ctx::int",
                "ctx::str",
            ]
        else:
            # check presence & type of non-context is_null columns
            assert is_string_dtype(ctx_data["ctx::non_ctx_1_id.non_ctx._is_null"])
            assert is_string_dtype(ctx_data["ctx::non_ctx_2_id.non_ctx._is_null"])
            assert is_string_dtype(tgt_data["non_ctx_1_id.non_ctx._is_null"])
            assert is_string_dtype(tgt_data["non_ctx_2_id.ctx._is_null"])
        # check pulled meta data
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        if model_type == "TABULAR":
            assert ctx_enctypes["ctx::non_ctx_2_id.non_ctx._is_null"] == ModelEncodingType.tabular_categorical
            assert tgt_enctypes["non_ctx_1_id.non_ctx._is_null"] == ModelEncodingType.tabular_categorical
        assert all([c in ctx_data.columns for c in ctx_enctypes.keys()])
        assert all([c in tgt_data.columns for c in tgt_enctypes.keys()])


class TestPullSCPinNonContextSetup(DisableMaskKeys):
    """
      a     b
     / \\   / \
    a1  ab1   b1
    where there is non-context connection between ab1 and b
    """

    def create_table_schema(self, path):
        a_path = Path(path) / "a.parquet"
        b_path = Path(path) / "b.parquet"
        a1_path = Path(path) / "a1.parquet"
        b1_path = Path(path) / "b1.parquet"
        ab1_path = Path(path) / "ab1.parquet"

        pd.DataFrame({"id": ["1", "2"], "cat": ["c1", "c2"]}).to_parquet(a_path)
        pd.DataFrame({"id": ["11", "12"], "cat": ["c11", "c12"]}).to_parquet(b_path)
        pd.DataFrame(
            {
                "a_id": ["1", "2", "2"],
                "aa_id": ["1", "1", "2"],
                "cat": ["c1", "c2", "c2"],
                "num": [1, 2, 2],
            }
        ).to_parquet(a1_path)
        pd.DataFrame(
            {
                "b_id": ["11", "11", "12", "12"],
                "cat": ["c1", "c1", "c2", "c2"],
                "num": [1, 1, 2, 2],
            }
        ).to_parquet(b1_path)
        pd.DataFrame(
            {
                "a_id": ["1", "1", "1", "2", "2", "2"],
                "b_id": ["11", "12", "13", "11", "12", "13"],
                "cat": ["c1", "c1", "c1", "c2", "c2", "c2"],
                "num": [11, 12, 13, 11, 12, 13],
            }
        ).to_parquet(ab1_path)

        tables = {
            "a": ParquetDataTable(path=a_path, primary_key="id", name="a"),
            "b": ParquetDataTable(path=b_path, primary_key="id", name="b"),
            "a1": ParquetDataTable(
                path=a1_path,
                foreign_keys=[
                    ForeignKey(column="a_id", referenced_table="a", is_context=True),
                    ForeignKey(column="aa_id", referenced_table="a", is_context=False),
                ],
            ),
            "b1": ParquetDataTable(
                path=b1_path,
                foreign_keys=[ForeignKey(column="b_id", referenced_table="b", is_context=True)],
            ),
            "ab1": ParquetDataTable(
                path=ab1_path,
                foreign_keys=[
                    ForeignKey(column="a_id", referenced_table="a", is_context=True),
                    ForeignKey(column="b_id", referenced_table="b", is_context=False),
                ],
            ),
        }
        return Schema(tables=tables)

    @pytest.mark.parametrize("do_ctx_only", [True, False])
    def test_b1_ctx(self, tmp_path, do_ctx_only):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull_fn = pull_context if do_ctx_only else pull
        pull_fn(tgt="b1", schema=schema, workspace_dir=tmp_path)
        # check pulled context data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == ["b::id", "b::cat"]
        assert set(ctx_df["b::id"]) == {"11", "12"}

        row_1 = ctx_df[ctx_df["b::id"] == "11"].iloc[0]
        assert row_1["b::id"] == "11"
        assert row_1["b::cat"] == "c11"
        row_2 = ctx_df[ctx_df["b::id"] == "12"].iloc[0]
        assert row_2["b::id"] == "12"
        assert row_2["b::cat"] == "c12"

    @pytest.mark.parametrize("do_ctx_only", [True, False])
    def test_ab1_ctx(self, tmp_path, do_ctx_only):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull_fn = pull_context if do_ctx_only else pull
        pull_fn(tgt="ab1", schema=schema, workspace_dir=tmp_path)
        # check pulled context data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == [
            "a::id",
            "a::cat",
            "a1::a_id",
            "a1::aa_id",
            "a1::cat",
            "a1::num",
        ]
        assert set(ctx_df["a::id"]) == {"1", "2"}

        row_1 = pd.Series(
            {
                "a::id": "1",
                "a::cat": "c1",
                "a1::a_id": ["1"],
                "a1::aa_id": ["1"],
                "a1::cat": ["c1"],
                "a1::num": [1],
            }
        )
        assert_series_equal(ctx_df[ctx_df["a::id"] == "1"].iloc[0], row_1, check_names=False)
        row_2 = pd.Series(
            {
                "a::id": "2",
                "a::cat": "c2",
                "a1::a_id": ["2", "2"],
                "a1::aa_id": ["1", "2"],
                "a1::cat": ["c2", "c2"],
                "a1::num": [2, 2],
            }
        )
        assert_series_equal(ctx_df[ctx_df["a::id"] == "2"].iloc[0], row_2, check_names=False)


class TestPullThreeTableSCPSetup(DisableMaskKeys):
    """
       a
    a1   a2

    The aim of this test is to check if we deal correctly with encoding types
    and if we can merge SCP context that has keys of various types.
    """

    def create_table_schema(self, path, make_seq_keys=False):
        a_path = Path(path) / "a.parquet"
        a1_path = Path(path) / "a1.parquet"
        a2_path = Path(path) / "a2.parquet"

        c = int if make_seq_keys else str
        pd.DataFrame({"id": [c("1"), c("2")]}).to_parquet(a_path)
        pd.DataFrame(
            {
                "a_id": [c("1"), c("1")],
                "id": [c("3"), c("4")],
                "cat": ["cat", "cat"],
                "num": [1, 1],
                "unused": [1, 1],
            }
        ).to_parquet(a1_path)
        pd.DataFrame({"a_id": [c("1"), c("1")], "id": [c("5"), c("6")]}).to_parquet(a2_path)

        tables = {
            "a": ParquetDataTable(path=a_path, primary_key="id", name="a"),
            "a1": ParquetDataTable(
                path=a1_path,
                primary_key="id",
                name="a1",
                encoding_types={
                    "cat": ModelEncodingType.tabular_categorical,
                    "num": ModelEncodingType.tabular_numeric_auto,
                },
                columns=["a_id", "id", "cat", "num"],
                foreign_keys=[ForeignKey(column="a_id", referenced_table="a", is_context=True)],
            ),
            "a2": ParquetDataTable(
                path=a2_path,
                primary_key="id",
                name="a2",
                foreign_keys=[ForeignKey(column="a_id", referenced_table="a", is_context=True)],
            ),
        }
        return Schema(tables=tables)

    def test_encoding_types_respected(self, tmp_path):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull(tgt="a2", schema=schema, workspace_dir=tmp_path)
        # check pulled context data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == [
            "a::id",
            "a1::a_id",
            "a1::id",
            "a1::cat",
            "a1::num",
        ]
        ctx_row1 = ctx_df[ctx_df["a::id"] == "1"].reset_index(drop=True).loc[0, :]
        ctx_row2 = ctx_df[ctx_df["a::id"] == "2"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a::id"] == "1"
        assert np.array_equal(ctx_row1["a1::a_id"], ["1", "1"])
        assert np.array_equal(ctx_row1["a1::id"], ["3", "4"])
        assert np.array_equal(ctx_row1["a1::cat"], ["cat", "cat"])
        assert np.array_equal(ctx_row1["a1::num"], [1.0, 1.0])
        assert ctx_row2["a::id"] == "2"
        assert np.array_equal(ctx_row2["a1::a_id"], [])
        assert np.array_equal(ctx_row2["a1::id"], [])
        assert np.array_equal(ctx_row2["a1::cat"], [])
        assert np.array_equal(ctx_row2["a1::num"], [])
        # check encoding types
        ctx_encoding_types = json.loads((tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json").read_text())
        assert ctx_encoding_types == {
            "a1::cat": ModelEncodingType.tabular_categorical,
            "a1::num": ModelEncodingType.tabular_numeric_auto,
        }

    def test_sequential_keys_work_for_pull_tgt_and_ctx(self, tmp_path):
        schema = self.create_table_schema(tmp_path, make_seq_keys=True)
        # pull data
        pull(tgt="a2", schema=schema, workspace_dir=tmp_path)
        # check pulled context data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        ctx_row1 = ctx_df[ctx_df["a::id"] == "1"].reset_index(drop=True).loc[0, :]
        ctx_row2 = ctx_df[ctx_df["a::id"] == "2"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a::id"] == "1"
        assert np.array_equal(ctx_row1["a1::a_id"], ["1", "1"])
        assert np.array_equal(ctx_row1["a1::id"], ["3", "4"])
        assert np.array_equal(ctx_row1["a1::cat"], ["cat", "cat"])
        assert np.array_equal(ctx_row1["a1::num"], [1.0, 1.0])
        assert ctx_row2["a::id"] == "2"
        assert np.array_equal(ctx_row2["a1::a_id"], [])
        assert np.array_equal(ctx_row2["a1::id"], [])
        assert np.array_equal(ctx_row2["a1::cat"], [])
        assert np.array_equal(ctx_row2["a1::num"], [])
        # check encoding types
        ctx_encoding_types = json.loads((tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json").read_text())
        assert ctx_encoding_types == {
            "a1::cat": ModelEncodingType.tabular_categorical,
            "a1::num": ModelEncodingType.tabular_numeric_auto,
        }

    def test_sequential_keys_work_for_pull_ctx_only(self, tmp_path):
        schema = self.create_table_schema(tmp_path, make_seq_keys=False)
        # pull data
        pull_context(tgt="a2", schema=schema, workspace_dir=tmp_path)
        # check pulled context data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        pull_context(tgt="a2", schema=schema, workspace_dir=tmp_path)
        ctx_row1 = ctx_df[ctx_df["a::id"] == "1"].reset_index(drop=True).loc[0, :]
        ctx_row2 = ctx_df[ctx_df["a::id"] == "2"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a::id"] == "1"
        assert np.array_equal(ctx_row1["a1::a_id"], ["1", "1"])
        assert np.array_equal(ctx_row1["a1::id"], ["3", "4"])
        assert np.array_equal(ctx_row1["a1::cat"], ["cat", "cat"])
        assert np.array_equal(ctx_row1["a1::num"], [1.0, 1.0])
        assert ctx_row2["a::id"] == "2"
        assert np.array_equal(ctx_row2["a1::a_id"], [])
        assert np.array_equal(ctx_row2["a1::id"], [])
        assert np.array_equal(ctx_row2["a1::cat"], [])
        assert np.array_equal(ctx_row2["a1::num"], [])


class TestPullComplexSCPSetup(DisableMaskKeys):
    """
                     a
     a1             a2              a3
    a11      a21   a22   a23        a31

    The aim of this test is to check if we properly build context in
    more complex case.
    """

    def create_table_schema(self, path):
        a_path = Path(path) / "a.parquet"
        a1_path = Path(path) / "a1.parquet"
        a11_path = Path(path) / "a11.parquet"
        a2_path = Path(path) / "a2.parquet"
        a21_path = Path(path) / "a21.parquet"
        a22_path = Path(path) / "a22.parquet"
        a23_path = Path(path) / "a23.parquet"
        a3_path = Path(path) / "a3.parquet"
        a31_path = Path(path) / "a31.parquet"

        pd.DataFrame({"id": ["a/1", "a/2"]}).to_parquet(a_path)
        pd.DataFrame({"a_id": ["a/1", "a/1"], "id": ["a1/3", "a1/4"]}).to_parquet(a1_path)
        pd.DataFrame({"a1_id": ["a1/3", "a1/3"], "id": ["a11/5", "a11/6"]}).to_parquet(a11_path)
        pd.DataFrame({"a_id": ["a/1", "a/1"], "id": ["a2/7", "a2/8"]}).to_parquet(a2_path)
        pd.DataFrame({"a2_id": ["a2/7", "a2/7"], "id": ["a21/9", "a21/10"]}).to_parquet(a21_path)
        pd.DataFrame({"a2_id": ["a2/7", "a2/7"], "id": ["a22/11", "a22/12"]}).to_parquet(a22_path)
        pd.DataFrame({"a2_id": ["a2/7", "a2/7"], "id": ["a23/13", "a23/14"]}).to_parquet(a23_path)
        pd.DataFrame({"a_id": ["a/1", "a/1"], "id": ["a3/15", "a3/16"]}).to_parquet(a3_path)
        pd.DataFrame({"a3_id": ["a3/15", "a3/15"], "id": ["a31/17", "a31/18"]}).to_parquet(a31_path)

        tables = {
            "a": ParquetDataTable(path=a_path, primary_key="id", name="a"),
            "a1": ParquetDataTable(
                path=a1_path,
                primary_key="id",
                name="a1",
                foreign_keys=[ForeignKey(column="a_id", referenced_table="a", is_context=True)],
            ),
            "a11": ParquetDataTable(
                path=a11_path,
                primary_key="id",
                name="a11",
                foreign_keys=[ForeignKey(column="a1_id", referenced_table="a1", is_context=True)],
            ),
            "a2": ParquetDataTable(
                path=a2_path,
                primary_key="id",
                name="a2",
                foreign_keys=[ForeignKey(column="a_id", referenced_table="a", is_context=True)],
            ),
            "a21": ParquetDataTable(
                path=a21_path,
                primary_key="id",
                name="a21",
                foreign_keys=[ForeignKey(column="a2_id", referenced_table="a2", is_context=True)],
            ),
            "a22": ParquetDataTable(
                path=a22_path,
                primary_key="id",
                name="a22",
                foreign_keys=[ForeignKey(column="a2_id", referenced_table="a2", is_context=True)],
            ),
            "a23": ParquetDataTable(
                path=a23_path,
                primary_key="id",
                name="a23",
                foreign_keys=[ForeignKey(column="a2_id", referenced_table="a2", is_context=True)],
            ),
            "a3": ParquetDataTable(
                path=a3_path,
                primary_key="id",
                name="a3",
                foreign_keys=[ForeignKey(column="a_id", referenced_table="a", is_context=True)],
            ),
            "a31": ParquetDataTable(
                path=a31_path,
                primary_key="id",
                name="a31",
                foreign_keys=[ForeignKey(column="a3_id", referenced_table="a3", is_context=True)],
            ),
        }
        return Schema(tables=tables)

    def test_pull_tgt_and_ctx(self, tmp_path):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull(tgt="a22", schema=schema, workspace_dir=tmp_path)
        # check pulled context data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == [
            "a2::a_id",
            "a2::id",
            "a::id",
            "a21::a2_id",
            "a21::id",
            "a1::a_id",
            "a1::id",
        ]
        ctx_row1 = ctx_df[ctx_df["a2::id"] == "a2/7"].reset_index(drop=True).loc[0, :]
        ctx_row2 = ctx_df[ctx_df["a2::id"] == "a2/8"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a2::a_id"] == "a/1"
        assert ctx_row1["a2::id"] == "a2/7"
        assert ctx_row1["a::id"] == "a/1"
        assert np.array_equal(ctx_row1["a21::a2_id"], ["a2/7", "a2/7"])
        assert np.array_equal(ctx_row1["a21::id"], ["a21/9", "a21/10"])
        assert np.array_equal(ctx_row1["a1::a_id"], ["a/1", "a/1"])
        assert np.array_equal(ctx_row1["a1::id"], ["a1/3", "a1/4"])
        assert ctx_row2["a2::a_id"] == "a/1"
        assert ctx_row2["a2::id"] == "a2/8"
        assert ctx_row2["a::id"] == "a/1"
        assert np.array_equal(ctx_row2["a21::a2_id"], [])
        assert np.array_equal(ctx_row2["a21::id"], [])
        assert np.array_equal(ctx_row2["a1::a_id"], ["a/1", "a/1"])
        assert np.array_equal(ctx_row2["a1::id"], ["a1/3", "a1/4"])

    def test_pull_tgt_and_ctx_language_model(self, tmp_path):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull(tgt="a22", schema=schema, model_type="LANGUAGE", workspace_dir=tmp_path)
        # check pulled data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == [
            "a2::a_id",
            "a2::id",
            "a::id",
        ]
        ctx_row1 = ctx_df[ctx_df["a2::id"] == "a2/7"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a2::a_id"] == "a/1"
        assert ctx_row1["a2::id"] == "a2/7"
        assert ctx_row1["a::id"] == "a/1"

    def test_pull_ctx_only(self, tmp_path):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull_context(tgt="a22", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == [
            "a2::a_id",
            "a2::id",
            "a::id",
            "a21::a2_id",
            "a21::id",
            "a1::a_id",
            "a1::id",
        ]
        ctx_row1 = ctx_df[ctx_df["a2::id"] == "a2/7"].reset_index(drop=True).loc[0, :]
        ctx_row2 = ctx_df[ctx_df["a2::id"] == "a2/8"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a2::a_id"] == "a/1"
        assert ctx_row1["a2::id"] == "a2/7"
        assert ctx_row1["a::id"] == "a/1"
        assert np.array_equal(ctx_row1["a21::a2_id"], ["a2/7", "a2/7"])
        assert np.array_equal(ctx_row1["a21::id"], ["a21/9", "a21/10"])
        assert np.array_equal(ctx_row1["a1::a_id"], ["a/1", "a/1"])
        assert np.array_equal(ctx_row1["a1::id"], ["a1/3", "a1/4"])
        assert ctx_row2["a2::a_id"] == "a/1"
        assert ctx_row2["a2::id"] == "a2/8"
        assert ctx_row2["a::id"] == "a/1"
        assert np.array_equal(ctx_row2["a21::a2_id"], [])
        assert np.array_equal(ctx_row2["a21::id"], [])
        assert np.array_equal(ctx_row2["a1::a_id"], ["a/1", "a/1"])
        assert np.array_equal(ctx_row2["a1::id"], ["a1/3", "a1/4"])

    def test_pull_ctx_only_language_model(self, tmp_path):
        schema = self.create_table_schema(tmp_path)
        # pull data
        pull_context(
            tgt="a22",
            schema=schema,
            workspace_dir=tmp_path,
            model_type="LANGUAGE",
        )
        # check pulled data
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert list(ctx_df.columns) == [
            "a2::a_id",
            "a2::id",
            "a::id",
        ]
        ctx_row1 = ctx_df[ctx_df["a2::id"] == "a2/7"].reset_index(drop=True).loc[0, :]
        ctx_row2 = ctx_df[ctx_df["a2::id"] == "a2/8"].reset_index(drop=True).loc[0, :]
        assert ctx_row1["a2::a_id"] == "a/1"
        assert ctx_row1["a2::id"] == "a2/7"
        assert ctx_row1["a::id"] == "a/1"
        assert ctx_row2["a2::a_id"] == "a/1"
        assert ctx_row2["a2::id"] == "a2/8"
        assert ctx_row2["a::id"] == "a/1"


class TestLanguageText:
    @pytest.fixture
    def single_table_with_pk(self, tmp_path):
        table_path = tmp_path / "tbl.parquet"
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "int": [10, 20, 30],
                "text1": ["first text", "second text", "third text"],
                "cat": ["a", "b", "a"],
                "text2": ["hello", "world", pd.NA],
            }
        ).to_parquet(table_path)
        table = ParquetDataTable(path=table_path, name="table.csv", primary_key="id")
        table.encoding_types |= {"text1": ModelEncodingType.language_text}
        table.encoding_types |= {"text2": ModelEncodingType.language_text}
        schema = Schema(tables={"table.csv": table})
        return schema

    @pytest.mark.parametrize("has_pk", [True, False])
    def test_pull_single_table_non_text_cols(self, tmp_path, single_table_with_pk, has_pk):
        schema = single_table_with_pk
        if not has_pk:
            schema.tables["table.csv"].primary_key = None
        # pull TABULAR columns
        pull(tgt="table.csv", schema=schema, workspace_dir=tmp_path)
        # check pulled data
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(tgt_data) == 3
        assert set(tgt_data.columns) == {
            "id",
            "int",
            "cat",
        }
        # checked pulled meta data
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        if has_pk:
            assert set(tgt_enctypes.keys()) == {"int", "cat"}
            assert tgt_keys == {"primary_key": "id"}
        else:
            assert set(tgt_enctypes.keys()) == {
                "id",
                "int",
                "cat",
            }
            assert tgt_keys == {}

    @pytest.mark.parametrize("has_pk", [True, False])
    def test_pull_single_table_text_cols(self, tmp_path, single_table_with_pk, has_pk):
        schema = single_table_with_pk
        if not has_pk:
            schema.tables["table.csv"].primary_key = None
        # pull LANGUAGE columns
        pull(
            tgt="table.csv",
            schema=schema,
            model_type="LANGUAGE",
            workspace_dir=tmp_path,
        )
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(ctx_data) == 3
        assert len(tgt_data) == 3
        assert set(tgt_data.columns) == {
            TEMPORARY_PRIMARY_KEY,
            "text1",
            "text2",
        }
        # checked pulled meta data
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        assert tgt_enctypes == {
            "text1": ModelEncodingType.language_text,
            "text2": ModelEncodingType.language_text,
        }
        ctx_keys = read_json(tmp_path / "OriginalData" / "ctx-meta" / "keys.json")
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        root_key = {"root_key": "table.csv::id"} if has_pk else {}
        assert ctx_keys == {"primary_key": f"table.csv::{TEMPORARY_PRIMARY_KEY}"} | root_key
        assert tgt_keys == {"context_key": TEMPORARY_PRIMARY_KEY}

        if has_pk:
            assert set(ctx_data.columns) == {
                f"table.csv::{TEMPORARY_PRIMARY_KEY}",
                "table.csv::id",
                "table.csv::int",
                "table.csv::cat",
            }
            assert ctx_enctypes == {
                "table.csv::int": ModelEncodingType.tabular_numeric_auto,
                "table.csv::cat": ModelEncodingType.tabular_categorical,
            }
        else:
            assert set(ctx_data.columns) == {
                f"table.csv::{TEMPORARY_PRIMARY_KEY}",
                "table.csv::id",
                "table.csv::int",
                "table.csv::cat",
            }
            assert ctx_enctypes == {
                "table.csv::id": ModelEncodingType.tabular_numeric_auto,
                "table.csv::int": ModelEncodingType.tabular_numeric_auto,
                "table.csv::cat": ModelEncodingType.tabular_categorical,
            }

    @pytest.mark.parametrize("has_pk", [True, False])
    def test_pull_single_table_ctx_only(self, tmp_path, single_table_with_pk, has_pk):
        table_path = tmp_path / "tbl.parquet"
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "int": [10, 20, 30],
                "cat": ["a", "b", "a"],
            }
        ).to_parquet(table_path)
        table = ParquetDataTable(path=table_path, name="table.csv", primary_key="id")
        table.encoding_types |= {"text1": ModelEncodingType.language_text}
        table.encoding_types |= {"text2": ModelEncodingType.language_text}
        schema = Schema(tables={"table.csv": table})

        if not has_pk:
            schema.tables["table.csv"].primary_key = None
        # pull LANGUAGE columns
        with patch(f"{PULL_MODULE}.determine_n_partitions", return_value=2):
            pull_context(
                tgt="table.csv",
                schema=schema,
                workspace_dir=tmp_path,
                model_type="LANGUAGE",
            )
        # check pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_data) == 3
        assert set(ctx_data.columns) == {
            f"table.csv::{TEMPORARY_PRIMARY_KEY}",
            "table.csv::id",
            "table.csv::int",
            "table.csv::cat",
        }

    @pytest.fixture
    def two_table_text(self, tmp_path):
        ctx_path = Path(tmp_path) / "ctx.parquet"
        tgt_path = Path(tmp_path) / "tgt.parquet"

        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "int": [0, 1, 99],
                "str": ["a", "b", "c"],
                "text1": ["one", "two", "three"],
            }
        ).to_parquet(ctx_path)

        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "ctx_id": [1, 2, 3],
                "int": [10, 20, 30],
                "text1": ["first text", "second text", "third text"],
                "cat": ["a", "b", "a"],
                "text2": ["hello", "world", pd.NA],
            }
        ).to_parquet(tgt_path)

        tgt_table = ParquetDataTable(
            path=tgt_path,
            primary_key="id",
            name="tgt",
            foreign_keys=[ForeignKey(column="ctx_id", referenced_table="ctx", is_context=True)],
        )
        tgt_table.encoding_types |= {"text1": ModelEncodingType.language_text}
        tgt_table.encoding_types |= {"text2": ModelEncodingType.language_text}

        ctx_table = ParquetDataTable(path=ctx_path, primary_key="id", name="ctx")
        ctx_table.encoding_types |= {"text1": ModelEncodingType.language_text}

        tables = {"ctx": ctx_table, "tgt": tgt_table}
        return Schema(tables=tables)

    @pytest.mark.parametrize("has_pk", [True, False])
    def test_pull_two_table_non_text_cols(self, tmp_path, two_table_text, has_pk):
        schema = two_table_text
        if not has_pk:
            schema.tables["tgt"].primary_key = None
        # pull TABULAR columns
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # tgt
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(tgt_data) == 3
        assert set(tgt_data.columns) == {"id", "ctx_id", "int", "cat"}
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        if has_pk:
            assert set(tgt_enctypes.keys()) == {"int", "cat"}
            assert tgt_keys == {"primary_key": "id", "context_key": "ctx_id"}
        else:
            assert set(tgt_enctypes.keys()) == {
                "id",
                "int",
                "cat",
            }
            assert tgt_keys == {"context_key": "ctx_id"}

        # ctx
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_data) == 3
        assert set(ctx_data.columns) == {"ctx::id", "ctx::int", "ctx::str"}
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        ctx_keys = read_json(tmp_path / "OriginalData" / "ctx-meta" / "keys.json")
        assert set(ctx_enctypes.keys()) == {"ctx::int", "ctx::str"}
        assert ctx_keys == {"primary_key": "ctx::id", "root_key": "ctx::id"}

    @pytest.mark.parametrize("has_pk", [True, False])
    def test_pull_two_table_text_cols(self, tmp_path, two_table_text, has_pk):
        schema = two_table_text
        if not has_pk:
            schema.tables["tgt"].primary_key = None
        # pull LANGUAGE columns
        pull(tgt="tgt", schema=schema, model_type="LANGUAGE", workspace_dir=tmp_path)
        # tgt
        pk = TEMPORARY_PRIMARY_KEY
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert len(tgt_data) == 3
        assert set(tgt_data.columns) == {pk, "text1", "text2"}
        tgt_enctypes = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        assert set(tgt_enctypes.keys()) == {"text1", "text2"}
        assert tgt_keys == {"context_key": pk}
        # ctx
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_data) == 3
        assert set(ctx_data.columns) == {
            "ctx::id",
            "ctx::int",
            "ctx::str",
            f"tgt::{pk}",
            "tgt::ctx_id",
            "tgt::id",
            "tgt::int",
            "tgt::cat",
        }
        ctx_keys = read_json(tmp_path / "OriginalData" / "ctx-meta" / "keys.json")
        assert ctx_keys == {"primary_key": f"tgt::{pk}", "root_key": "ctx::id"}
        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        ctx_enctypes_expected = {"ctx::int", "ctx::str", "tgt::int", "tgt::cat"}
        if not has_pk:
            ctx_enctypes_expected |= {"tgt::id"}
        assert set(ctx_enctypes.keys()) == ctx_enctypes_expected

    @pytest.mark.parametrize("has_pk", [True, False])
    def test_pull_two_table_ctx_only(self, tmp_path, two_table_text, has_pk):
        ctx_path = Path(tmp_path) / "ctx.parquet"
        tgt_path = Path(tmp_path) / "tgt.parquet"
        pd.DataFrame(
            {
                "ctx::id": [1, 2, 3],
                "ctx::int": [0, 1, 99],
                "ctx::str": ["a", "b", "c"],
                "ctx::text1": ["one", "two", "three"],
            }
        ).to_parquet(ctx_path)
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "ctx_id": [1, 2, 3],
                "int": [10, 20, 30],
                "cat": ["a", "b", "a"],
            }
        ).to_parquet(tgt_path)

        tgt_table = ParquetDataTable(
            path=tgt_path,
            primary_key="id",
            name="tgt",
            foreign_keys=[ForeignKey(column="ctx_id", referenced_table="ctx", is_context=True)],
        )
        tgt_table.encoding_types |= {"text1": ModelEncodingType.language_text}
        tgt_table.encoding_types |= {"text2": ModelEncodingType.language_text}

        ctx_table = ParquetDataTable(path=ctx_path, primary_key="ctx::id", name="ctx")
        ctx_table.encoding_types |= {"ctx::text1": ModelEncodingType.language_text}

        tables = {"ctx": ctx_table, "tgt": tgt_table}

        schema = Schema(tables=tables)

        if not has_pk:
            schema.tables["tgt"].primary_key = None
        # pull LANGUAGE columns
        pull_context(
            tgt="tgt",
            schema=schema,
            model_type="LANGUAGE",
            workspace_dir=tmp_path,
        )
        # tgt
        pk = TEMPORARY_PRIMARY_KEY
        # ctx
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_data) == 3
        assert set(ctx_data.columns) == {
            "ctx::id",
            "ctx::int",
            "ctx::str",
            f"tgt::{pk}",
            "tgt::id",
            "tgt::ctx_id",
            "tgt::int",
            "tgt::cat",
        }

    def test_ctx_only_on_empty_tabular_data(self, tmp_path):
        df = pd.DataFrame(index=range(100))
        fn = tmp_path / "data.parquet"
        df.to_parquet(fn, index=True)

        tables = {"tgt": ParquetDataTable(path=fn, encoding_types={"text": ModelEncodingType.language_text})}
        schema = Schema(tables=tables)

        pull_context(tgt="tgt", schema=schema, workspace_dir=tmp_path, model_type="LANGUAGE")

        assert (tmp_path / "OriginalData" / "ctx-data").exists()
        assert not (tmp_path / "OriginalData" / "tgt-data").exists()
        ctx_df = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert len(ctx_df) == 100


class TestPullWithSequentialContext:
    @pytest.fixture
    def three_table_data(self):
        # 10 grandparent context records
        # each with 10 context records
        # half of those with 10 target records
        gpc_df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "str": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        )
        ctx_df = pd.DataFrame(
            {
                "gpc_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,
                "id": list(range(100)),
                "dt": pd.date_range(start="1/1/2000", periods=50).to_list() + [None] * 50,
                "int": pd.Series([1, 2, 3, None] * 25, dtype="int64[pyarrow]"),
            },
        )
        tgt_df = pd.DataFrame(
            {
                "ctx_id": list(range(50)) * 10,
                "id": list(range(500)),
                "int": list(range(500)),
                "str": ["o"] * 500,
            }
        )
        return gpc_df, ctx_df, tgt_df

    @pytest.fixture
    def three_table_syn_data(self, three_table_data):
        gpc_df, ctx_df, tgt_df = three_table_data
        gpc_df = gpc_df.add_prefix("gpc::")
        ctx_df = ctx_df.add_prefix("ctx::")
        tgt_df = tgt_df.add_prefix("")
        return gpc_df, ctx_df, tgt_df

    def test_pull(self, tmp_path, three_table_data):
        gpc_df, ctx_df, tgt_df = three_table_data
        schema = create_three_table_schema(tmp_path, gpc_df, ctx_df, tgt_df, tgt_pk="id")
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check nested sequences in the pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert ctx_data.columns.to_list() == [
            "ctx::gpc_id",
            "ctx::id",
            "ctx$prev::dt",
            "ctx::dt",
            "ctx$prev::int",
            "ctx::int",
            "gpc::id",
            "gpc::str",
        ]
        # check nested sequences in pulled meta data
        assert pd.api.types.is_object_dtype(ctx_data.dtypes["ctx$prev::dt"])
        assert pd.api.types.is_object_dtype(ctx_data.dtypes["ctx$prev::int"])

        def check_equal_length(row):
            return len({len(x) for x in row}) <= 1

        # ensure both arrays per each row are of equal size
        assert bool(ctx_data[["ctx$prev::dt", "ctx$prev::int"]].apply(check_equal_length, axis=1).all())

        ctx_enctypes = read_json(tmp_path / "OriginalData" / "ctx-meta" / "encoding-types.json")
        assert len(ctx_enctypes) == 5  # 3 regular, 2 NS
        assert ctx_enctypes["ctx$prev::dt"] == ModelEncodingType.tabular_datetime

    def test_pull_syn(self, tmp_path, three_table_syn_data):
        gpc_df, ctx_df, tgt_df = three_table_syn_data
        schema = create_three_table_schema(
            tmp_path,
            gpc_df,
            ctx_df,
            tgt_df,
            gpc_pk="gpc::id",
            ctx_fk="ctx::gpc_id",
            ctx_pk="ctx::id",
            tgt_fk="ctx_id",
            tgt_pk="id",
        )
        pull_context(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        # check nested sequences in the pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert ctx_data.columns.tolist() == [
            "ctx::gpc_id",
            "ctx::id",
            "ctx$prev::dt",
            "ctx::dt",
            "ctx$prev::int",
            "ctx::int",
            "gpc::id",
            "gpc::str",
        ]

    @pytest.fixture
    def uoi(self):
        n = 100  # number of users

        # Users DataFrame
        df_users = pd.DataFrame(
            {
                "id": range(1, n + 1),
            }
        )

        # Orders DataFrame
        df_orders = pd.DataFrame(
            {
                "id": range(1, n * 30 + 1),  # Each user has 30 orders
                "user_id": np.repeat(df_users["id"].values, 30),  # Repeat each user_id 30 times
            }
        )
        start_date = pd.to_datetime("2022-01-01")
        end_date = pd.to_datetime("2022-12-31")
        # Creating a total order time period in days
        total_days = (end_date - start_date).days

        # Create random fractions for each user (each user has 30 fractions in ascending order)
        random_days = np.sort(np.random.rand(n, 30), axis=1) * total_days

        # Create the datetime series for all orders
        df_orders["placed"] = (start_date + pd.to_timedelta(random_days.flatten(), unit="D")).astype("<M8[us]")

        # Items DataFrame
        df_items = pd.DataFrame(
            {
                "id": range(1, n * 90 + 1),  # Each order has 3 items
                "order_id": np.repeat(df_orders["id"].values, 3),  # Repeat each order_id 3 times
                "product": [
                    (i % 9) + 1 for i in range(n * 90)
                ],  # Product cycles through 1-9 for each group of 3 orders
            }
        )

        return df_users, df_orders, df_items

    def test_pull_uoi(self, tmp_path, uoi):
        gpc_df, ctx_df, tgt_df = uoi
        schema = create_three_table_schema(
            tmp_path,
            gpc_df,
            ctx_df,
            tgt_df,
            gpc_name="user",
            ctx_name="order",
            tgt_name="item",
            ctx_fk="user_id",
            tgt_fk="order_id",
            tgt_pk="id",
        )
        keys, _ = mask_keys(key_columns=["user_id"], ctx_data=ctx_df[["user_id"]])
        pull(
            tgt="item",
            schema=schema,
            workspace_dir=tmp_path,
        )
        # check nested sequences in the pulled data
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        ns_cols = [col for col in ctx_data.columns if "$" in col]
        assert ns_cols == ["order$prev::placed"]
        for col in ns_cols:
            assert bool(
                ctx_data[keys].groupby("user::id")[col].apply(lambda x: x.str.len().is_monotonic_increasing).all()
            )

            # check that all values are arrays
            assert ctx_data[col].apply(lambda x: isinstance(x, (list, np.ndarray))).all()


class TestSpecialCases:
    @staticmethod
    def create_col_pk_and_fk_schema(path):
        path = Path(path)
        # prepare DFs and write to disk
        n = 10
        pks = [str(i) for i in range(n)]
        # case 1: multiple roles for a key (PK & FK)
        non_ctx_df = pd.DataFrame({"id": pks, "str": [f"str_{i}" for i in range(n)]})
        tgt_df = pd.DataFrame({"id": pks, "num": range(n)})
        ext_df = pd.DataFrame({"id": pks, "dt": [None] * n})
        # case 2: table and sequence table with PK name = FK name
        tbl_id_df = pd.DataFrame({"id": pks, "col": ["a"] * n})
        tbl_df = pd.DataFrame({"id": pks * 2, "num": range(n * 2)})
        # save all to files
        non_ctx_df.to_csv(path / "non_ctx.csv", index=False)
        tgt_df.to_csv(path / "tgt.csv", index=False)
        ext_df.to_csv(path / "ext.csv", index=False)
        tbl_id_df.to_csv(path / "tbl_id.csv", index=False)
        tbl_df.to_csv(path / "tbl.csv", index=False)

        # prepare schema with foreign keys
        tables = {
            # case 1
            "non_ctx": CsvDataTable(
                path=path / "non_ctx.csv",
                name="non_ctx",
                primary_key="id",
                columns=["id", "str"],
            ),
            "tgt": CsvDataTable(
                path=path / "tgt.csv",
                name="tgt",
                primary_key="id",
                columns=["id", "num"],
                foreign_keys=[ForeignKey(column="id", referenced_table="non_ctx", is_context=False)],
            ),
            "ext": CsvDataTable(
                path=path / "ext.csv",
                name="ext",
                primary_key="id",
                columns=["id", "dt"],
                foreign_keys=[ForeignKey(column="id", referenced_table="tgt", is_context=True)],
            ),
            # case 2
            "tbl_id": CsvDataTable(
                path=path / "tbl_id.csv",
                name="tbl_id",
                primary_key="id",
                columns=["id", "col"],
            ),
            "tbl": CsvDataTable(
                path=path / "tbl.csv",
                name="tbl",
                columns=["id", "num"],
                foreign_keys=[ForeignKey(column="id", referenced_table="tbl_id", is_context=True)],
            ),
        }

        return Schema(tables=tables)

    @pytest.mark.parametrize(
        "pull_table,pk",
        [("tgt", None), ("ext", "id"), ("tbl_id", "id"), ("tbl", None)],
    )
    def test_pull_col_pk_and_fk(self, tmp_path, pull_table, pk):
        schema = self.create_col_pk_and_fk_schema(tmp_path)
        pull(tgt=pull_table, schema=schema, workspace_dir=tmp_path)
        tgt_keys = read_json(tmp_path / "OriginalData" / "tgt-meta" / "keys.json")
        if pk:
            assert tgt_keys == {"primary_key": pk}
        else:
            assert "primary_key" not in tgt_keys


class TestPullAtScale:
    def test_multiple_chunks_order(self, tmp_path):
        tgt, ctx = "tgt", "ctx"
        ctx_path = tmp_path / f"{ctx}.parquet"
        tgt_path = tmp_path / f"{tgt}.parquet"
        pk = "id"
        col = "int"
        n = 20_000
        seqlen = 200
        ctx_df = pd.DataFrame({pk: list(range(n)), col: list(range(n))})
        tgt_df = pd.DataFrame({pk: list(range(n)) * seqlen, col: list(range(n * seqlen))})
        ctx_df.to_parquet(ctx_path)
        tgt_df.to_parquet(tgt_path)

        tables = {
            ctx: ParquetDataTable(path=ctx_path, primary_key=pk, name=ctx),
            tgt: ParquetDataTable(
                path=tgt_path,
                primary_key=pk,
                name=tgt,
                foreign_keys=[ForeignKey(column=pk, referenced_table=ctx, is_context=True)],
            ),
        }

        schema = Schema(tables=tables)
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)

        # check that pulled ctx is shuffled
        ctx_data = pd.read_parquet(tmp_path / "OriginalData" / "ctx-data")
        assert ctx_data[f"{ctx}::{col}"].is_monotonic_increasing is False

        # check that pulled tgt keeps the order of the original data
        # NOTE: sequences are long enough to ensure multiple chunks, multiple partitions
        #       the test mainly intends to check if the order is preserved after consolidation of chunks into partitions
        tgt_data = pd.read_parquet(tmp_path / "OriginalData" / "tgt-data")
        assert sum(tgt_data.groupby(pk)[col].apply(lambda s: s.is_monotonic_increasing)) == n


class TestPullEmptySequences:
    def test_pull_empty_sequences(self, tmp_path):
        ctx_path = tmp_path / "ctx.parquet"
        tgt_path = tmp_path / "tgt.parquet"
        ctx_df = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        tgt_df = pd.DataFrame({"id": [10]})
        ctx_df.to_parquet(ctx_path)
        tgt_df.to_parquet(tgt_path)

        tables = {
            "ctx": ParquetDataTable(path=ctx_path, primary_key="id", name="ctx"),
            "tgt": ParquetDataTable(
                path=tgt_path,
                primary_key="id",
                name="tgt",
                foreign_keys=[ForeignKey(column="id", referenced_table="ctx", is_context=True)],
            ),
        }

        schema = Schema(tables=tables)

        # check that all partition files have been created, and match between ctx and tgt
        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path)
        ctx_files = sorted([f.name for f in (tmp_path / "OriginalData" / "ctx-data").glob("*.parquet")])
        tgt_files = sorted([f.name for f in (tmp_path / "OriginalData" / "tgt-data").glob("*.parquet")])
        assert ctx_files == tgt_files

        pull(tgt="tgt", schema=schema, workspace_dir=tmp_path, max_sample_size=1)

        # check that all partition files have been created
        ctx_files = sorted([f.name for f in (tmp_path / "OriginalData" / "ctx-data").glob("*.parquet")])
        tgt_files = sorted([f.name for f in (tmp_path / "OriginalData" / "tgt-data").glob("*.parquet")])
        assert ctx_files == tgt_files
