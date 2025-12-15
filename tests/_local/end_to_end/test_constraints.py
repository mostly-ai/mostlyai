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

"""simple e2e test for constraints feature."""

import numpy as np
import pandas as pd
import pytest

from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import FixedCombination, Inequality


@pytest.fixture(scope="module")
def mostly(tmp_path_factory):
    """create local MostlyAI instance for testing."""
    yield MostlyAI(local=True, local_dir=tmp_path_factory.mktemp("mostlyai"), quiet=True)


def test_constraints_with_partial_seed(mostly):
    """test FixedCombination and Inequality constraints with partial seed data."""

    # create training data with both constraint types
    df = pd.DataFrame(
        {
            "state": ["CA", "NY", "TX"] * 30,
            "city": ["LA", "NYC", "Houston"] * 30,
            "start_age": [20, 25, 30] * 30,
            "end_age": [25, 30, 35] * 30,
            "value": np.random.rand(90),
        }
    )

    # define valid state-city pairs
    valid_pairs = {("CA", "LA"), ("NY", "NYC"), ("TX", "Houston")}

    # train generator with both constraints
    g = mostly.train(
        config={
            "name": "Test Constraints E2E",
            "tables": [
                {
                    "name": "test",
                    "data": df,
                    "tabular_model_configuration": {
                        "max_epochs": 0.5,
                    },
                }
            ],
            "constraints": [
                FixedCombination(table_name="test", columns=["state", "city"]),
                Inequality(table_name="test", low_column="start_age", high_column="end_age"),
            ],
        }
    )

    # verify generator was created
    assert g is not None

    # test 1: generate without seed - verify constraints are satisfied
    sd = mostly.generate(g, size=50)
    df_syn = sd.data()

    # verify columns
    assert "state" in df_syn.columns
    assert "city" in df_syn.columns
    assert "start_age" in df_syn.columns
    assert "end_age" in df_syn.columns

    # verify FixedCombination constraint
    syn_pairs = set(zip(df_syn["state"], df_syn["city"]))
    assert syn_pairs.issubset(valid_pairs), f"invalid pairs found: {syn_pairs - valid_pairs}"

    # verify Inequality constraint
    assert (df_syn["start_age"] <= df_syn["end_age"]).all(), "inequality constraint violated"

    sd.delete()

    # test 2: generate with partial seed - test all variations
    seed_state = ["CA", "NY", None, "TX", "CA", None, "NY", None]
    seed_city = ["LA", None, "Houston", "Houston", None, "NYC", None, None]
    seed_start_age = [50, 45, 40, None, 35, 30, None, 25]
    seed_end_age = [None, 55, None, 50, 45, None, 40, None]

    seed_df = pd.DataFrame(
        {
            "state": seed_state,
            "city": seed_city,
            "start_age": seed_start_age,
            "end_age": seed_end_age,
        }
    )

    sd = mostly.generate(g, seed=seed_df, size=30)
    df_syn = sd.data()

    # verify seed values appear in same order
    for i, row in df_syn.head(len(seed_df)).iterrows():
        if seed_state[i] is not None:
            assert row["state"] == seed_state[i]
        if seed_city[i] is not None:
            assert row["city"] == seed_city[i]
        if seed_start_age[i] is not None:
            assert row["start_age"] == seed_start_age[i]
        if seed_end_age[i] is not None:
            assert row["end_age"] == seed_end_age[i]

    assert (df_syn["start_age"] <= df_syn["end_age"]).all()

    # cleanup
    g.delete()
    sd.delete()


@pytest.mark.skip(reason="check what's wrong with language model constraints")
def test_language_model_constraints(mostly):
    """test constraints with language model encoding types (categorical, numeric, datetime)."""

    # create training data with language model columns
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"] * 30,  # will use LANGUAGE_CATEGORICAL
            "subcategory": ["X", "Y", "Z"] * 30,  # will use LANGUAGE_CATEGORICAL
            "min_value": [10, 20, 30] * 30,  # will use LANGUAGE_NUMERIC
            "max_value": [15, 25, 35] * 30,  # will use LANGUAGE_NUMERIC
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"] * 30),  # will use LANGUAGE_DATETIME
            "end_date": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"] * 30),  # will use LANGUAGE_DATETIME
        }
    )

    # define valid category-subcategory pairs
    valid_pairs = {("A", "X"), ("B", "Y"), ("C", "Z")}

    # train generator with language model and constraints
    g = mostly.train(
        config={
            "name": "Test Language Model Constraints",
            "tables": [
                {
                    "name": "test",
                    "data": df,
                    "language_model_configuration": {
                        "max_epochs": 0.5,
                    },
                    "columns": [
                        {"name": "category", "model_encoding_type": "LANGUAGE_CATEGORICAL"},
                        {"name": "subcategory", "model_encoding_type": "LANGUAGE_CATEGORICAL"},
                        {"name": "min_value", "model_encoding_type": "LANGUAGE_NUMERIC"},
                        {"name": "max_value", "model_encoding_type": "LANGUAGE_NUMERIC"},
                        {"name": "start_date", "model_encoding_type": "LANGUAGE_DATETIME"},
                        {"name": "end_date", "model_encoding_type": "LANGUAGE_DATETIME"},
                    ],
                }
            ],
            "constraints": [
                FixedCombination(table_name="test", columns=["category", "subcategory"]),
                Inequality(table_name="test", low_column="min_value", high_column="max_value"),
                Inequality(table_name="test", low_column="start_date", high_column="end_date"),
            ],
        }
    )

    # verify generator was created
    assert g is not None

    # generate synthetic data
    sd = mostly.generate(g, size=50)
    df_syn = sd.data()

    # verify synthetic data has correct columns
    assert "category" in df_syn.columns
    assert "subcategory" in df_syn.columns
    assert "min_value" in df_syn.columns
    assert "max_value" in df_syn.columns
    assert "start_date" in df_syn.columns
    assert "end_date" in df_syn.columns

    # verify fixed combination constraint (category-subcategory pairs)
    syn_pairs = set(zip(df_syn["category"], df_syn["subcategory"]))
    assert syn_pairs.issubset(valid_pairs), f"invalid pairs found: {syn_pairs - valid_pairs}"

    # verify inequality constraint for numeric columns
    assert (df_syn["min_value"] <= df_syn["max_value"]).all(), "numeric inequality constraint violated"

    # verify inequality constraint for datetime columns
    df_syn["start_date"] = pd.to_datetime(df_syn["start_date"])
    df_syn["end_date"] = pd.to_datetime(df_syn["end_date"])
    assert (df_syn["start_date"] <= df_syn["end_date"]).all(), "datetime inequality constraint violated"

    # cleanup
    g.delete()
    sd.delete()
