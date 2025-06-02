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

import base64
import csv
import io
import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd

warnings.simplefilter("always", DeprecationWarning)


def convert_to_base64(
    df: pd.DataFrame | list[dict[str, Any]],
    format: Literal["parquet", "jsonl"] = "parquet",
) -> str:
    """
    Convert a DataFrame to a base64 encoded string, representing the content in Parquet or JSONL format.

    Args:
        df: The DataFrame to convert.
        format: The format to use for the conversion. Either "parquet" or "jsonl".

    Returns:
        The base64 encoded string.
    """
    if df.__class__.__name__ == "DataFrame" and df.__class__.__module__.startswith("pyspark.sql"):
        # Convert PySpark DataFrame to Pandas DataFrame (safely)
        df = pd.DataFrame(df.collect(), columns=df.columns)
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # Save the DataFrame to a buffer in Parquet / JSONL format
    buffer = io.BytesIO()
    if format == "parquet":
        # clear any (potentially non-serializable) attributes that might stop us from saving to PQT
        if df.attrs:
            df.attrs.clear()
        # persist the DataFrame to Parquet format
        df.to_parquet(buffer, index=False)
    else:  # format == "jsonl"
        # persist the DataFrame to JSONL format
        df.to_json(buffer, orient="records", date_format="iso", lines=True, index=False)
    # read in persisted file as base64 encoded string
    buffer.seek(0)
    binary_data = buffer.read()
    base64_encoded_str = base64.b64encode(binary_data).decode()
    return base64_encoded_str


def convert_to_df(data: str, format: Literal["parquet", "jsonl"] = "parquet") -> pd.DataFrame:
    # Load the DataFrame from a base64 encoded string
    binary_data = base64.b64decode(data)
    buffer = io.BytesIO(binary_data)
    if format == "parquet":
        df = pd.read_parquet(buffer)
    else:  # format == "jsonl"
        df = pd.read_json(buffer, orient="records", lines=True)
    return df


def read_table_from_path(path: str | Path) -> (str, pd.DataFrame):
    # read data from file
    fn = str(path)
    if fn.lower().endswith((".pqt", ".parquet")):
        df = pd.read_parquet(fn)
    else:
        delimiter = ","
        if fn.lower().endswith((".csv", ".tsv")):
            try:
                with open(fn) as f:
                    header = f.readline()
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(header, ",;|\t' :").delimiter
            except (csv.Error, FileNotFoundError):
                # csv.Error: happens for example for single column CSV files
                # FileNotFoundError: happens for example for remote files
                pass
        df = pd.read_csv(fn, low_memory=False, delimiter=delimiter)
    if fn.lower().endswith((".gz", ".gzip", ".bz2")):
        fn = fn.rsplit(".", 1)[0]
    name = Path(fn).stem
    return name, df
