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

# %%
#!pip install -U mostlyai
import pandas as pd

from mostlyai.sdk import MostlyAI

mostly = MostlyAI()

url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/baseball/players.csv.gz"
df = pd.read_csv(url).head(1000)
df["birthYear"] = (pd.to_datetime(df.birthDate).dt.year).astype("Int64")
df["deathYear"] = (pd.to_datetime(df.deathDate).dt.year).astype("Int64")
df = df[["birthDate", "deathDate", "birthYear", "deathYear", "throws"]]

g = mostly.train(
    config={
        "tables": [
            {
                "name": "players",
                "data": df,
            }
        ],
        "constraints": [
            {
                "type": "Inequality",
                "config": {"table_name": "players", "low_column": "birthYear", "high_column": "deathYear"},
            },
            {
                "type": "Inequality",
                "config": {"table_name": "players", "low_column": "birthDate", "high_column": "deathDate"},
            },
        ],
    },
)

syn = mostly.probe(g, size=1000)
assert (syn.birthYear < syn.deathYear).mean() == 1.0
assert (syn.birthDate < syn.deathDate).mean() == 1.0


# %%
pd.to_datetime(df.deathDate).max()

# %%
