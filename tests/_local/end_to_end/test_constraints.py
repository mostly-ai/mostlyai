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
from mostlyai.sdk.domain import ConstraintConfig


@pytest.fixture(scope="module")
def mostly(tmp_path_factory):
    """create local MostlyAI instance for testing."""
    yield MostlyAI(local=True, local_dir=tmp_path_factory.mktemp("mostlyai"), quiet=True)


def test_constraints(mostly):
    """test FixedCombinations and Inequality constraints with flights-like data."""

    # create training data with both constraint types
    # valid route-airline combinations (similar to flights dataset)
    valid_triplets = [("JFK", "LAX", "AA"), ("LAX", "ORD", "UA"), ("ORD", "JFK", "DL")]

    # generate air_time first, then ensure elapsed_time = air_time + ground_time
    air_time = np.random.randint(60, 300, 90)  # 1-5 hours in minutes
    ground_time = np.random.randint(20, 60, 90)  # 20-60 min ground time
    elapsed_time = air_time + ground_time  # ensure air_time < elapsed_time

    # generate departure times, then ensure arrival_time = departure_time + flight_duration
    departure_time = pd.date_range(start="2024-01-01 08:00", periods=90, freq="3h")
    flight_duration = pd.Timedelta(hours=2) + pd.to_timedelta(np.random.randint(0, 60, 90), unit="m")
    arrival_time = departure_time + flight_duration  # ensure departure_time < arrival_time

    df = pd.DataFrame(
        {
            "ORIGIN_AIRPORT": [t[0] for t in valid_triplets] * 30,
            "DESTINATION_AIRPORT": [t[1] for t in valid_triplets] * 30,
            "AIRLINE": [t[2] for t in valid_triplets] * 30,
            # numeric inequality: air_time < elapsed_time (air_time + taxi/ground time)
            "AIR_TIME": air_time,
            "ELAPSED_TIME": elapsed_time,
            # datetime inequality: departure < arrival (strict boundaries)
            "DEPARTURE_TIME": departure_time,
            "ARRIVAL_TIME": arrival_time,
        }
    )

    # define expected time difference range (2-3 hours based on training data)
    min_time_diff = pd.Timedelta(hours=2)
    max_time_diff = pd.Timedelta(hours=3)
    expected_mean_time_diff = pd.Timedelta(hours=2.5)  # midpoint of 2-3 hours

    # define valid origin-destination-airline triplets
    valid_combos = {("JFK", "LAX", "AA"), ("LAX", "ORD", "UA"), ("ORD", "JFK", "DL")}

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
                # both snake_case and camelCase are supported for the config dict
                ConstraintConfig(
                    type="FixedCombinations",
                    config={
                        "table_name": "test",
                        "columns": ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"],
                    },
                ),
                ConstraintConfig(
                    type="Inequality",
                    config={"table_name": "test", "low_column": "AIR_TIME", "high_column": "ELAPSED_TIME"},
                ),
                ConstraintConfig(
                    type="Inequality",
                    config={
                        "tableName": "test",
                        "lowColumn": "DEPARTURE_TIME",
                        "highColumn": "ARRIVAL_TIME",
                    },
                ),
            ],
        }
    )

    # generate synthetic data
    sd = mostly.generate(g, size=50)
    df_syn = sd.data()

    # verify FixedCombinations constraint
    syn_triplets = set(zip(df_syn["ORIGIN_AIRPORT"], df_syn["DESTINATION_AIRPORT"], df_syn["AIRLINE"]))
    assert syn_triplets.issubset(valid_combos), f"invalid triplets found: {syn_triplets - valid_combos}"

    # verify numeric Inequality constraint
    assert (df_syn["AIR_TIME"] <= df_syn["ELAPSED_TIME"]).all(), (
        "inequality constraint violated: AIR_TIME must be <= ELAPSED_TIME"
    )

    # verify datetime Inequality constraint
    assert (df_syn["DEPARTURE_TIME"] <= df_syn["ARRIVAL_TIME"]).all(), (
        "datetime inequality constraint violated: DEPARTURE_TIME must be <= ARRIVAL_TIME"
    )

    # verify time differences follow predefined rules
    time_diffs = df_syn["ARRIVAL_TIME"] - df_syn["DEPARTURE_TIME"]
    assert (time_diffs >= min_time_diff).all(), (
        f"time difference too small: min={time_diffs.min()}, expected >= {min_time_diff}"
    )
    assert (time_diffs <= max_time_diff).all(), (
        f"time difference too large: max={time_diffs.max()}, expected <= {max_time_diff}"
    )
    # verify overall mean time difference is close to expected value
    assert np.abs(time_diffs.mean() - expected_mean_time_diff) < pd.Timedelta(minutes=12), (
        f"overall mean time difference is not close to {expected_mean_time_diff}: mean={time_diffs.mean()}, expected ≈ {expected_mean_time_diff}"
    )

    g.delete()
    sd.delete()
