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

from unittest.mock import Mock

import pandas as pd
import pytest
from mostlyai.engine._workspace import Workspace

from mostlyai.sdk._local.execution.step_generate_model_report_data import execute_step_generate_model_report_data
from mostlyai.sdk.domain import ModelType


@pytest.mark.parametrize("null_split", ["trn", "val"])
def test_report_context_with_all_null_split(tmp_path, monkeypatch, null_split):
    workspace = Workspace(tmp_path)
    workspace.tgt_stats.write({})
    workspace.ctx_stats.write({"keys": {"primary_key": "id"}})
    workspace.ctx_data_path.mkdir(parents=True, exist_ok=True)
    for split, ids in [("trn", [1, 2]), ("val", [3, 4])]:
        pd.DataFrame({"id": ids, "value": [None, None] if split == null_split else ["a", "b"]}).to_parquet(
            workspace.ctx_data_path / f"part.000000-{split}.parquet"
        )
    monkeypatch.setattr(
        "mostlyai.sdk._local.execution.step_generate_model_report_data.qa_sample_size_heuristic", lambda **kwargs: 4
    )
    generate = Mock()
    monkeypatch.setattr("mostlyai.engine.generate", generate)

    execute_step_generate_model_report_data(
        workspace_dir=tmp_path, model_type=ModelType.tabular, update_progress=Mock()
    )

    context = generate.call_args.kwargs["ctx_data"].sort_values("id")
    assert context["id"].tolist() == [1, 2, 3, 4]
    assert context["value"].isna().sum() == 2
    assert context["value"].dropna().tolist() == ["a", "b"]
