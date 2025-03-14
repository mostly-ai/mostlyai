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

import re
import logging
import time

import pandas as pd

from mostlyai.engine._common import safe_convert_datetime
from mostlyai.sdk.domain import ModelEncodingType
from mostlyai.sdk._data.base import DataTable
from mostlyai.sdk._data.dtype import VirtualDType, VirtualVarchar, VirtualInteger
from mostlyai.sdk._data.util.common import run_with_timeout_unsafe, absorb_errors

AUTODETECT_SAMPLE_SIZE = 10_000
AUTODETECT_TIMEOUT = 15
LAT_LONG_REGEX = re.compile(r"^\s*-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?\s*$")
PK_POSSIBLE_VIRTUAL_DTYPES = (VirtualVarchar, VirtualInteger)

_LOG = logging.getLogger(__name__)


def auto_detect_encoding_types_and_pk(
    table: DataTable, initial_auto_encoding_types: dict[str, ModelEncodingType] | None = None
) -> tuple[dict, str | None]:
    if initial_auto_encoding_types is None:
        initial_auto_encoding_types = {col: ModelEncodingType.auto for col in table.columns}
    # sub-select only the columns which got the default categorical encoding type
    column_candidates = [
        c
        for c, enc in table.encoding_types.items()
        if enc in (ModelEncodingType.tabular_categorical, ModelEncodingType.language_categorical)
    ]
    # sub-select primary key candidates before sampling the data
    dtypes = VirtualDType.from_dtypes(table.dtypes)
    primary_key_candidates = [
        c
        for c, t in dtypes.items()
        if type(t) in PK_POSSIBLE_VIRTUAL_DTYPES
        and c.lower().endswith("id")
        and initial_auto_encoding_types[c] == ModelEncodingType.auto
    ]
    columns_to_sample = [c for c in table.columns if c in primary_key_candidates or c in column_candidates]
    fallback = (
        {
            c: ModelEncodingType.language_text
            if initial_auto_encoding_types[c] == ModelEncodingType.language_auto
            else ModelEncodingType.tabular_categorical
            for c in column_candidates
        },
        None,
    )

    def auto_detection_logic():
        return_vals = None
        t0 = time.time()
        with absorb_errors():
            data_sample = next(table.read_chunks(columns=columns_to_sample, fetch_chunk_size=AUTODETECT_SAMPLE_SIZE))
            primary_key = auto_detect_primary_key(data_sample[primary_key_candidates])
            # continue auto-detection for column_candidates and the remaining primary_key_candidates that are not numeric
            remaining_columns_to_auto_detect = [
                c for c in columns_to_sample if c != primary_key and type(dtypes[c]) is not VirtualInteger
            ]
            # auto-detect encoding types for the sampled data
            return_vals = (
                {
                    c: auto_detect_encoding_type(data_sample[c], initial_auto_encoding_types[c])
                    for c in remaining_columns_to_auto_detect
                },
                primary_key,
            )
        _LOG.info(f"auto_detect_encoding_types_and_pk logic for table={table.name} took {time.time() - t0:.2f} seconds")
        return return_vals if return_vals is not None else fallback

    # wrap the detection logic with timeout and fallback
    return run_with_timeout_unsafe(
        auto_detection_logic,
        timeout=AUTODETECT_TIMEOUT,
        fallback=fallback,
    )


def auto_detect_primary_key(sample: pd.DataFrame) -> str | None:
    # assuming sample columns are (1) ending with "id" and (2) of PK_POSSIBLE_VIRTUAL_DTYPES dtype
    for c in sample.columns:
        # check (3) all values unique and non-null and (4) of max len 36 (e.g. uuid len)
        if sample[c].is_unique and sample[c].notnull().all() and sample[c].astype(str).str.len().max() <= 36:
            return c
    return None


def auto_detect_encoding_type(x: pd.Series, initial_auto_encoding_type: ModelEncodingType | None = None):
    x = x.dropna()
    x = x.astype(str)
    x = x[x.str.strip() != ""]  # filter out empty and whitespace-only strings

    # if all non-null values can be converted to datetime -> datetime encoding
    if safe_convert_datetime(x).notna().all():
        return (
            ModelEncodingType.language_datetime
            if initial_auto_encoding_type == ModelEncodingType.language_auto
            else ModelEncodingType.tabular_datetime
        )

    # if all values match lat/long pattern -> lat_long (geo) encoding
    if x.str.match(LAT_LONG_REGEX).all():
        return (
            ModelEncodingType.language_text
            if initial_auto_encoding_type == ModelEncodingType.language_auto
            else ModelEncodingType.tabular_lat_long
        )

    # if more than 5% of rows contain unique values
    # TABULAR_AUTO -> TABULAR_CHARACTER
    # LANGUAGE_AUTO -> LANGUAGE_TEXT
    # AUTO -> TABULAR_CHARACTER if all values have the same length, otherwise LANGUAGE_TEXT
    if len(x.dropna()) >= 100 and x.value_counts().eq(1).reindex(x).mean() > 0.05:
        return (
            ModelEncodingType.language_text
            if initial_auto_encoding_type == ModelEncodingType.language_auto
            or (initial_auto_encoding_type == ModelEncodingType.auto and x.str.len().nunique() > 1)
            else ModelEncodingType.tabular_character
        )

    # fallback to categorical
    return (
        ModelEncodingType.language_categorical
        if initial_auto_encoding_type == ModelEncodingType.language_auto
        else ModelEncodingType.tabular_categorical
    )
