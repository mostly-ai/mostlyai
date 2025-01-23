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

import os
from pathlib import Path

import pytest

from mostlyai.sdk._data.file.table.csv import CsvDataTable


SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
CSV_FIXTURES_DIR = SCRIPT_DIR / "file" / "fixtures" / "csv"


@pytest.fixture
def user_table():
    user_path = CSV_FIXTURES_DIR / "order_management" / "User.csv"
    return CsvDataTable(path=user_path)


@pytest.fixture
def user_df(user_table):
    return user_table.read_data()
