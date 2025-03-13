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

from mostlyai.sdk import MostlyAI
import pandas as pd


def test_connector(tmp_path):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    c = mostly.connect(
        config={
            "name": "Test 1",
            "type": "S3_STORAGE",
            "access_type": "SOURCE",
            "config": {
                "access_key": "XXX",
            },
            "secrets": {
                "secret_key": "YYY",
            },
        },
        test_connection=False,
    )
    assert c.name == "Test 1"
    c.update(name="Test 2", test_connection=False)
    assert c.name == "Test 2"

    c.delete()


def test_read_data(tmp_path):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    # Create a temporary CSV file
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({"num": [1, 2, 3, 4, 5, 6], "let": ["a", "b", "c", "d", "e", "f"]})
    df.to_csv(csv_file, index=False)

    # Create a connector for the CSV file
    c = mostly.connect(
        config={
            "name": "Local CSV Connector",
            "type": "FILE_UPLOAD",
            "access_type": "READ_DATA",
            "config": {},
            "secrets": {},
        },
        test_connection=False,
    )

    read_df = c.read_data(location=str(csv_file))
    pd.testing.assert_frame_equal(read_df, df, check_dtype=False)

    limited_df = c.read_data(location=str(csv_file), limit=3)
    pd.testing.assert_frame_equal(limited_df, df.head(3), check_dtype=False)

    shuffled_df = c.read_data(location=str(csv_file), shuffle=True)
    assert not shuffled_df.equals(df), "Data should be shuffled"
    assert set(shuffled_df["num"]) == set(df["num"]), "Shuffled data should contain the same elements"
    assert set(shuffled_df["let"]) == set(df["let"]), "Shuffled data should contain the same elements"

    limited_shuffled_df = c.read_data(location=str(csv_file), limit=3, shuffle=True)
    assert len(limited_shuffled_df) == 3, "Limited shuffled data should have 3 rows"
    assert set(limited_shuffled_df["num"]).issubset(set(df["num"])), (
        "Limited shuffled data should contain a subset of elements"
    )
    assert set(limited_shuffled_df["let"]).issubset(set(df["let"])), (
        "Limited shuffled data should contain a subset of elements"
    )

    c.delete()
