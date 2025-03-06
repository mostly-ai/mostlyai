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

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_csv_file(tmp_path):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "bool": [True, False, pd.NA, np.nan],
            "int": [1212, 2512, pd.NA, np.nan],
            "float": [0.1, -1.0, pd.NA, np.nan],
            "date": ["2020-04-04", "1877-05-05", pd.NA, np.nan],
            "ts_s": ["2020-04-04 14:14:14", "1877-05-05 01:01:01", pd.NA, np.nan],
            "ts_ns": ["2020-04-04 14:14:14.44", "1877-05-05 01:01:01.11", pd.NA, np.nan],
            "ts_tz": ["1999-09-15T09:37:50.871127Z", "1998-02-19T08:12:02.573302Z", pd.NA, np.nan],
            "text": ['This is a "quoted" text', "Row 日本", pd.NA, np.nan],
            "bigint": [1372636854620000520, 1372637091620000337, pd.NA, np.nan],
        }
    )
    fn = tmp_path / "sample.csv"
    df.to_csv(fn, index=False)
    return fn


@pytest.fixture
def sample_parquet_file(tmp_path, sample_csv_file, request):
    df = pd.read_csv(sample_csv_file, engine="pyarrow", dtype_backend=request.param)
    fn = tmp_path / f"sample_{request.param}.parquet"
    df.to_parquet(fn, engine="pyarrow")
    return fn
