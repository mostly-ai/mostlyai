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

from pathlib import Path

from mostlyai.engine._workspace import Workspace


def migrate_workspace(workspace_dir: Path) -> None:
    workspace = Workspace(workspace_dir)
    # migrate min5/max5 in column stats to min/max (<= 4.5.6)
    for stats_pathdesc in [workspace.ctx_stats, workspace.tgt_stats]:
        stats = stats_pathdesc.read()
        if stats:
            for col, col_stats in stats.get("columns", {}).items():
                if col_stats.get("min5") is not None:
                    col_stats["min"] = min(col_stats["min5"]) if len(col_stats["min5"]) > 0 else None
                    col_stats.pop("min5")
                if col_stats.get("max5") is not None:
                    col_stats["max"] = max(col_stats["max5"]) if len(col_stats["max5"]) > 0 else None
                    col_stats.pop("max5")
            stats_pathdesc.write(stats)
