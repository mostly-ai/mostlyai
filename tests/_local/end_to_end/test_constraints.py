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

    g.delete()
    sd.delete()
