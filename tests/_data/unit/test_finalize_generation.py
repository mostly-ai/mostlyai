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

import math
import zipfile

import numpy as np
import pandas as pd
import pytest

from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.dtype import STRING
from mostlyai.sdk._data.file.table.csv import CsvDataTable
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._local.execution.jobs import _merge_tabular_language_data
from mostlyai.sdk._local.execution.step_finalize_generation import (
    export_data_to_excel,
    finalize_table_generation,
    zip_data,
)


@pytest.fixture
def tgt_data():
    yield pd.DataFrame(
        {
            "gender": ["female", "male"] * 3,
            "age": [45, 23, 61, 19, 33, 49],
        }
    )


def test_finalize_table_generation(tmp_path, tgt_data):
    tgt_data.to_parquet(tmp_path / "tgt.parquet")
    tgt_table = ParquetDataTable(path=tmp_path / "tgt.parquet", name="tgt")

    gen_data = tgt_table.read_data()
    gen_data_path = tmp_path / "gen"
    gen_data_path.mkdir(exist_ok=True, parents=True)
    # make 3 partitions
    get_data_partitioned = np.array_split(gen_data, 3)
    # save each part into a separate parquet file
    for i, df_part in enumerate(get_data_partitioned):
        part_fn = gen_data_path / f"part.{i:06}.parquet"
        df_part.to_parquet(part_fn, index=False)

    gen_table = ParquetDataTable(path=gen_data_path, columns=tgt_table.columns, name="tgt")
    schema = Schema(
        tables={"tgt": gen_table},
    )
    # post-process
    finalize_table_generation(
        generated_data_schema=schema,
        target_table_name="tgt",
        delivery_dir=tmp_path,
        export_csv=True,
        job_workspace_dir=tmp_path,
    )
    # check post-processed data
    df_expected = tgt_data
    post_pqt_path = tmp_path / "tgt" / "parquet"
    post_pqt = ParquetDataTable(path=post_pqt_path).read_data()
    pd.testing.assert_frame_equal(post_pqt, df_expected, check_dtype=False)
    assert post_pqt["gender"].dtype == STRING
    post_csv = CsvDataTable(path=tmp_path / "tgt" / "csv" / "tgt.csv").read_data()
    pd.testing.assert_frame_equal(post_csv, df_expected, check_dtype=False)
    assert post_csv["gender"].dtype == STRING


def test_export_data_to_excel(tmp_path):
    # prepare parquet files
    delivery_dir = tmp_path / "FinalizedSyntheticData"
    tables = ["A", "B", "C"]
    for table in tables:
        pqt_dir = delivery_dir / table / "parquet"
        pqt_dir.mkdir(parents=True, exist_ok=True)
        for fidx in range(3):
            offset = 3 * fidx
            pd.DataFrame(
                {
                    "A": [offset, offset + 1, offset + 2],
                    "X": ["x"] * 3,
                    "N": ["", pd.NA, None],
                    # "D": pd.to_datetime(['2024-02-03', '2024-04-05', pd.NA]),
                    # "DT": pd.to_datetime(['2024-02-03 15:43', '2024-04-05 02:32', pd.NA]),
                }
            ).to_parquet(f"{pqt_dir}/part.{fidx:06}.parquet")

    export_data_to_excel(delivery_dir=delivery_dir, output_dir=tmp_path)

    # Read the Excel file
    excel_file_path = tmp_path / "synthetic-samples.xlsx"
    xls = pd.ExcelFile(excel_file_path)

    expected_values = pd.DataFrame({"A": [0, 1, 2, 3, 4, 5, 6, 7, 8], "X": ["x"] * 9, "N": [None] * 9})
    # Iterate over each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if sheet_name == "_TOC_":
            df_orig = pd.DataFrame({"Table": tables})
            # assert ignoring the rows with Notes
            pd.testing.assert_frame_equal(df[["Table"]][0 : len(df_orig)], df_orig)
        else:
            # Check the expected values
            df.sort_values(by="A", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            pd.testing.assert_frame_equal(df, expected_values, check_dtype=False)

    # Check for the sheet names
    expected_sheets = ["_TOC_", "A", "B", "C"]
    assert sorted(xls.sheet_names) == sorted(expected_sheets)


def test_zip_data(tmp_path):
    # write parquet files to FinalizedSyntheticData
    delivery_dir = tmp_path / "FinalizedSyntheticData"
    for tidx in range(10):
        dir = delivery_dir / f"table_{tidx}" / "parquet"
        dir.mkdir(parents=True, exist_ok=True)
        for fidx in range(5):
            (dir / f"part.{fidx:06}.parquet").touch()
    # execute zip_data
    zip_dir = tmp_path / "ZIP"
    zip_data(delivery_dir=delivery_dir, format="parquet", out_dir=zip_dir)
    # count zipped files
    with zipfile.ZipFile(zip_dir / "synthetic-parquet-data.zip", "r") as zipf:
        n_zipped_files = len(zipf.namelist())
    assert n_zipped_files == 10 * 5  # 10 tables, 5 partitions each


def test_merge_tabular_language_data(tmp_path):
    ctx_dir = tmp_path / "OriginalData" / "ctx-data"
    tgt_dir = tmp_path / "SyntheticData"
    ctx_dir.mkdir(parents=True)
    tgt_dir.mkdir(parents=True)
    n_merged_part = 10
    n_rows = 6 * 8 * 2
    n_tgt_part = 8
    n_ctx_part = 6
    n_rows_per_tgt_part = n_rows // n_tgt_part
    n_rows_per_ctx_part = n_rows // n_ctx_part
    for i in range(n_tgt_part):
        tgt_fn = tgt_dir / f"part.{i:06}.{0:06}.parquet"
        tgt_df = pd.DataFrame([(i, "pk")] * n_rows_per_tgt_part, columns=["text", "__primary_key"])
        tgt_df.to_parquet(tgt_fn)
    for i in range(n_ctx_part):
        ctx_fn = ctx_dir / f"part.{i:06}.{0:06}.parquet"
        ctx_df = pd.DataFrame(
            [(i, i, "pk")] * n_rows_per_ctx_part,
            columns=["ctx::col", "tgt::non_text", "tgt::__primary_key"],
        )
        ctx_df.to_parquet(ctx_fn)
    _merge_tabular_language_data(
        workspace_dir=tmp_path,
        merged_part_size=n_merged_part,
    )
    out_fns = sorted(list(tgt_dir.glob("*.parquet")))
    assert len(out_fns) == math.ceil(n_rows / n_merged_part)
    out_df = pd.concat([pd.read_parquet(fn) for fn in out_fns])
    assert out_df.columns.tolist() == ["non_text", "text"]
    assert out_df["text"].is_monotonic_increasing
    assert out_df["non_text"].is_monotonic_increasing
