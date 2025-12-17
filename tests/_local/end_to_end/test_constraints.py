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


def test_constraints(mostly):
    """test FixedCombination and Inequality constraints with partial seed data."""

    # create training data with both constraint types
    df = pd.DataFrame(
        {
            "state": ["CA", "NY", "TX"] * 30,
            "city": ["LA", "NYC", "Houston"] * 30,
            "start_age": [20, 25, 30] * 30,
            "end_age": [25, 30, 35] * 30,
            # start_date varies every 6 hours, end_date is 1-30 days after start_date
            "start_date": pd.date_range(start="2024-01-01", periods=90, freq="6h"),
            "end_date": pd.date_range(start="2024-01-01", periods=90, freq="6h")
            + pd.Timedelta(days=1)
            + pd.to_timedelta(np.random.randint(0, 30, 90), unit="d"),
            "value": np.random.rand(90),
        }
    )

    # define expected time difference range (1-30 days based on training data)
    min_time_diff = pd.Timedelta(days=1)
    max_time_diff = pd.Timedelta(days=30)

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
                Inequality(table_name="test", low_column="start_date", high_column="end_date", strict_boundaries=True),
            ],
        }
    )

    # generate synthetic data
    sd = mostly.generate(g, size=50)
    df_syn = sd.data()

    # verify FixedCombination constraint
    syn_pairs = set(zip(df_syn["state"], df_syn["city"]))
    assert syn_pairs.issubset(valid_pairs), f"invalid pairs found: {syn_pairs - valid_pairs}"

    # verify Inequality constraint
    assert (df_syn["start_age"] <= df_syn["end_age"]).all(), "inequality constraint violated"

    # verify datetime Inequality constraint (strict boundaries)
    assert (df_syn["start_date"] < df_syn["end_date"]).all(), (
        "datetime inequality constraint violated: start_date must be < end_date"
    )

    # verify time differences follow predefined rules
    time_diffs = df_syn["end_date"] - df_syn["start_date"]
    assert (time_diffs >= min_time_diff).all(), (
        f"time difference too small: min={time_diffs.min()}, expected >= {min_time_diff}"
    )
    assert (time_diffs <= max_time_diff).all(), (
        f"time difference too large: max={time_diffs.max()}, expected <= {max_time_diff}"
    )
    # verify overall mean time difference is close to 15 days
    assert np.abs(time_diffs.mean() - pd.Timedelta(days=15)) < pd.Timedelta(days=1), (
        f"overall mean time difference is not close to 15 days: mean={time_diffs.mean()}, expected ≈ {pd.Timedelta(days=15)}"
    )

    g.delete()
    sd.delete()
