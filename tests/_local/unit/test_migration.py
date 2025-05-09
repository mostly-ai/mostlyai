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

import pytest
from mostlyai.engine._workspace import Workspace
from mostlyai.sdk._local.execution.migration import migrate_workspace


@pytest.mark.parametrize(
    "ctx_stats, tgt_stats",
    [
        (
            None,
            {
                "columns": {
                    "datetime": {
                        "min5": [f"2024-01-0{i}" for i in range(1, 6)],
                        "max5": [f"2025-01-0{i}" for i in range(1, 6)],
                    }
                }
            },
        ),
        (
            {
                "columns": {
                    "int": {
                        "min5": [i for i in range(1, 6)],
                        "max5": [100 + i for i in range(1, 6)],
                    },
                    "float": {
                        "min5": [],
                        "max5": [],
                    },
                }
            },
            {
                "columns": {
                    "datetime": {
                        "min5": [f"2024-01-0{i}" for i in range(1, 6)],
                        "max5": [f"2025-01-0{i}" for i in range(1, 6)],
                    }
                }
            },
        ),
    ],
)
def test_migrate_workspace(tmp_path, ctx_stats, tgt_stats):
    workspace_dir = tmp_path / "ModelStore"
    workspace = Workspace(workspace_dir)
    if ctx_stats:
        workspace.ctx_stats.write(ctx_stats)
    workspace.tgt_stats.write(tgt_stats)
    migrate_workspace(workspace_dir)
    if ctx_stats:
        migrated_ctx_stats = workspace.ctx_stats.read()
        assert migrated_ctx_stats["columns"]["int"]["min"] == 1
        assert migrated_ctx_stats["columns"]["int"]["max"] == 105
        assert migrated_ctx_stats["columns"]["float"]["min"] is None
        assert migrated_ctx_stats["columns"]["float"]["max"] is None
    else:
        assert not workspace.ctx_stats.path.exists()
    migrated_tgt_stats = workspace.tgt_stats.read()
    assert migrated_tgt_stats["columns"]["datetime"]["min"] == "2024-01-01"
    assert migrated_tgt_stats["columns"]["datetime"]["max"] == "2025-01-05"
