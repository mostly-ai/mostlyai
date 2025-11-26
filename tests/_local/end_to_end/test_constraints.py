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
from mostlyai.sdk.domain import FixedCombination


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
                        "constraints": [FixedCombination(columns=["state", "city"])],
                    },
                }
            ],
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

    # NOTE: This is a POC test - the constraint logic is implemented but needs
    # further debugging to work correctly. The infrastructure is in place:
    # 1. FixedCombination constraint class
    # 2. Data transformation (merge/split columns)
    # 3. Preprocessing step to transform training data
    # 4. Reverse transformation during generation
    # 5. Column metadata handling
    # TODO: Debug why the model isn't learning the merged column correctly

    # verify data is generated
    assert len(df_syn) == 50
    assert "state" in df_syn.columns
    assert "city" in df_syn.columns

    # NOTE: Uncomment this when constraint logic is fully working:
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
