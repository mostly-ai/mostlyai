# Copyright 2024-2025 MOSTLY AI
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
import zipfile

import numpy as np
import pandas as pd

from mostlyai.sdk import MostlyAI


def test_multi_table_with_text(tmp_path):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    # create mock data
    players_df = pd.DataFrame(
        {
            "id": range(200),
            "cat": ["a", "b"] * 100,
            "text": ["c", "d"] * 100,
            "parent_id": np.random.randint(0, 200, 200),
        }
    )
    batting_df = pd.DataFrame(
        {
            "players_id": list(range(200)) * 4,
            "num": np.random.uniform(size=800),
        }
    )
    fielding_df = pd.DataFrame(
        {
            "players_id": list(range(200)) * 2,
            "date": pd.date_range(start="2000-01-01", end="2030-12-31", periods=400).to_series(),
            "text": ["a", "b"] * 200,
        }
    )

    # GENERATOR
    g = mostly.train(
        config={
            "tables": [
                {
                    "name": "players",
                    "data": players_df,
                    "primary_key": "id",
                    "foreign_keys": [
                        {"column": "parent_id", "referenced_table": "players", "is_context": False},
                    ],
                    "columns": [
                        {"name": "id"},
                        {"name": "cat"},
                        {"name": "text", "model_encoding_type": "LANGUAGE_TEXT"},
                        {"name": "parent_id"},
                    ],
                    "tabular_model_configuration": {
                        "max_epochs": 0.1,
                    },
                    "language_model_configuration": {
                        "max_epochs": 0.1,
                    },
                },
                {
                    "name": "batting",
                    "data": batting_df,
                    "primary_key": None,
                    "foreign_keys": [
                        {"column": "players_id", "referenced_table": "players", "is_context": True},
                    ],
                    "columns": [
                        {"name": "players_id"},
                        {"name": "num"},
                    ],
                    "tabular_model_configuration": {
                        "max_epochs": 0.1,
                    },
                },
                {
                    "name": "fielding",
                    "data": fielding_df,
                    "primary_key": None,
                    "foreign_keys": [
                        {"column": "players_id", "referenced_table": "players", "is_context": True},
                    ],
                    "columns": [
                        {"name": "players_id"},
                        {"name": "date"},
                        {"name": "text", "model_encoding_type": "LANGUAGE_TEXT"},
                    ],
                    "tabular_model_configuration": {
                        "max_epochs": 0.1,
                        "enable_model_report": False,
                    },
                    "language_model_configuration": {
                        "max_epochs": 0.1,
                        "enable_model_report": False,
                    },
                },
            ],
        }
    )
    assert g.tables[0].total_rows == 200
    assert g.tables[1].total_rows == 800
    assert g.tables[2].total_rows == 400
    cat_col = g.tables[0].columns[1]
    assert set(cat_col.value_range.values) == {"a", "b"}
    num_col = g.tables[1].columns[1]
    assert num_col.model_encoding_type == "TABULAR_NUMERIC_BINNED"
    dat_col = g.tables[2].columns[1]
    assert dat_col.model_encoding_type == "TABULAR_DATETIME"

    assert g.tables[0].tabular_model_metrics is not None
    assert g.tables[0].language_model_metrics is not None
    assert g.tables[1].tabular_model_metrics is not None
    assert g.tables[1].language_model_metrics is None
    # model report for fielding (both tabular and language) is disabled
    assert g.tables[2].tabular_model_metrics is None
    assert g.tables[2].language_model_metrics is None

    sd = mostly.generate(g, size=20)
    syn = sd.data()
    assert len(syn["players"]) == 20
    assert len(syn["batting"]) == 80
    assert len(syn["fielding"]) == 40
    assert sd.tables[0].configuration.enable_data_report  # players
    assert sd.tables[1].configuration.enable_data_report  # batting
    assert not sd.tables[2].configuration.enable_data_report  # fielding
    reports_zip_path = sd.reports(tmp_path)
    with zipfile.ZipFile(reports_zip_path, "r") as zip_ref:
        expected_files = {
            "players-tabular-data.html",
            "players-tabular.html",
            "players-language-data.html",
            "players-language.html",
            "batting-tabular-data.html",
            "batting-tabular.html",
        }
        assert set(zip_ref.namelist()) == expected_files

    syn = mostly.probe(g, seed=[{"cat": "a"}])
    assert syn["players"]["cat"][0] == "a"
    assert len(syn["batting"]) == 4
    assert len(syn["fielding"]) == 2
