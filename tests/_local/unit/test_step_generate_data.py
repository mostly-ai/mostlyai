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

from mostlyai.sdk._data.util.common import TABLE_COLUMN_INFIX, TEMPORARY_PRIMARY_KEY
from mostlyai.sdk._local.execution.jobs import _merge_tabular_language_data
from mostlyai.sdk._local.execution.step_generate_data import write_key_only_context


def test_write_key_only_context(tmp_path):
    write_key_only_context(workspace_dir=tmp_path, table_name="tbl", n_rows=7)
    files = list((tmp_path / "OriginalData" / "ctx-data").glob("*.parquet"))
    assert len(files) == 1
    ctx = pd.read_parquet(files[0])
    key = f"tbl{TABLE_COLUMN_INFIX}{TEMPORARY_PRIMARY_KEY}"
    assert ctx.columns.tolist() == [key]
    assert len(ctx) == 7
    assert ctx[key].is_unique


def test_key_only_context_merges_with_language_only_output(tmp_path):
    # a table with LANGUAGE columns only: the merge must yield just the generated text columns
    write_key_only_context(workspace_dir=tmp_path, table_name="tbl", n_rows=5)
    tgt_dir = tmp_path / "SyntheticData"
    tgt_dir.mkdir()
    pd.DataFrame({"text": list("abcde")}).to_parquet(tgt_dir / "part.000000.000000.parquet")
    _merge_tabular_language_data(workspace_dir=tmp_path)
    out = pd.concat([pd.read_parquet(f) for f in sorted(tgt_dir.glob("*.parquet"))])
    assert out.columns.tolist() == ["text"]
    assert out["text"].tolist() == list("abcde")
