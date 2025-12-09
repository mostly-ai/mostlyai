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


def test_fixed_combination_constraint(mostly):
    """test that fixed combination constraint preserves valid combinations."""

    # create training data with specific state-city pairs
    df = pd.DataFrame(
        {
            "state": ["CA", "NY", "TX"] * 30,
            "city": ["LA", "NYC", "Houston"] * 30,
            "value": np.random.rand(90),
        }
    )

    # define valid pairs that should be preserved
    valid_pairs = {("CA", "LA"), ("NY", "NYC"), ("TX", "Houston")}

    # train generator with fixed combination constraint
    g = mostly.train(
        config={
            "name": "Test Constraints",
            "tables": [
                {
                    "name": "test",
                    "data": df,
                    "tabular_model_configuration": {
                        "max_epochs": 0.5,
                    },
                }
            ],
            "constraints": [FixedCombination(table_name="test", columns=["state", "city"])],
        }
    )

    # verify generator was created
    assert g is not None
    assert g.name == "Test Constraints"

    # generate synthetic data
    sd = mostly.generate(g, size=50)
    df_syn = sd.data()

    # verify synthetic data has correct columns
    assert "state" in df_syn.columns
    assert "city" in df_syn.columns
    assert "value" in df_syn.columns

    # verify only valid (state, city) pairs exist in synthetic data
    syn_pairs = set(zip(df_syn["state"], df_syn["city"]))

    # current status: POC implementation complete, but model may not preserve
    # all constraint combinations perfectly. This assertion validates the core
    # constraint preservation functionality.
    assert syn_pairs.issubset(valid_pairs), f"invalid pairs found: {syn_pairs - valid_pairs}"

    # verify all valid pairs are represented (with high probability)
    # allow some variance due to randomness
    assert len(syn_pairs) >= 2, "at least 2 different combinations should be generated"

    # cleanup
    g.delete()
    sd.delete()


def test_no_constraints_regression(mostly):
    """test that tables without constraints work exactly as before."""

    # create simple training data
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3] * 10,
            "col2": [4, 5, 6] * 10,
        }
    )

    # train generator WITHOUT constraints
    g = mostly.train(
        config={
            "name": "Test No Constraints",
            "tables": [
                {
                    "name": "test",
                    "data": df,
                    "tabular_model_configuration": {"max_epochs": 0.5},
                }
            ],
        }
    )

    # verify generator was created
    assert g is not None

    # generate synthetic data
    sd = mostly.generate(g, size=20)
    df_syn = sd.data()

    # verify synthetic data has correct shape
    assert df_syn.shape[0] == 20
    assert df_syn.shape[1] == 2
    assert "col1" in df_syn.columns
    assert "col2" in df_syn.columns

    # cleanup
    g.delete()
    sd.delete()


def test_constraints_with_seed_data(mostly):
    """test that seed data values are preserved during constraint transformations."""

    # create training data with inequality constraint
    df = pd.DataFrame(
        {
            "start_age": [20, 25, 30, 35, 40] * 20,
            "end_age": [25, 30, 35, 40, 45] * 20,
            "value": np.random.rand(100),
        }
    )

    # train generator with inequality constraint
    g = mostly.train(
        config={
            "name": "Test Constraints with Seed",
            "tables": [
                {
                    "name": "test",
                    "data": df,
                    "tabular_model_configuration": {
                        "max_epochs": 0.5,
                    },
                }
            ],
            "constraints": [Inequality(table_name="test", low_column="start_age", high_column="end_age")],
        }
    )

    # verify generator was created
    assert g is not None

    # create seed data with specific values that should be preserved
    seed_df = pd.DataFrame(
        {
            "start_age": [50, 55],
            "end_age": [60, 65],  # both columns seeded - should be preserved
            "value": [999, 888],  # extra column
        }
    )

    # generate with seed data (size=None means use seed size)
    sd = mostly.generate(g, seed=seed_df)
    df_syn = sd.data()

    # verify synthetic data has correct columns
    assert "start_age" in df_syn.columns
    assert "end_age" in df_syn.columns
    assert "value" in df_syn.columns

    # verify seed values are preserved in the output
    # for subject tables with seed, first N rows should match seed (where N is seed size)
    # check if seed values appear in the output (they may be in first rows or scattered)
    seed_start_ages = {50, 55}
    seed_end_ages = {60, 65}
    syn_start_ages = set(df_syn["start_age"].unique())
    syn_end_ages = set(df_syn["end_age"].unique())

    # verify seed start_age values appear in output
    assert len(seed_start_ages & syn_start_ages) > 0, (
        f"seed start_age values not found. seed: {seed_start_ages}, syn: {syn_start_ages}"
    )

    # verify seed end_age values appear in output (this is the key test - end_age should be preserved)
    assert len(seed_end_ages & syn_end_ages) > 0, (
        f"seed end_age values not found. seed: {seed_end_ages}, syn: {syn_end_ages}"
    )

    # verify inequality constraint is satisfied for all rows
    assert (df_syn["start_age"] <= df_syn["end_age"]).all(), "inequality constraint violated"

    # cleanup
    g.delete()
    sd.delete()
