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

"""
Data push
"""

import logging
from pathlib import Path

import pandas as pd
from cloudpathlib import CloudPath

from mostlyai.sdk._data.base import DataTable, Schema
from mostlyai.sdk._data.dtype import PandasDType
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable

_LOG = logging.getLogger(__name__)


def push_data(
    source: ParquetDataTable,
    destination: DataTable,
    schema: Schema | None = None,
    overwrite_tables: bool = False,
) -> None:
    """
    Push data from source to destination, with respect to the provided schema (if applicable)
    """

    _LOG.info(f"push data from `{source}` to `{destination}`")

    source_files = source.dataset.files
    n_partitions = len(source_files)
    _LOG.info(f"PUSH will handle {n_partitions} partitions")

    if not schema:
        schema = Schema(tables={destination.name: destination})

    def partitions():
        for part_i, part_file in enumerate(source_files, start=1):
            part_table = ParquetDataTable(path=part_file)
            partition_text = f"partition {part_i} out of {n_partitions} partitions"
            _LOG.info(f"read {partition_text}")
            tgt_data = part_table.read_data()
            tgt_data = adapt_dtypes_to_destination(tgt_data=tgt_data, schema=schema, tgt=destination.name)
            _LOG.info(f"prepared {partition_text} for push")
            yield part_file, tgt_data

    # write DataFrame partitions to destination
    # Note: at the moment we are not yet adapting dtypes to destination
    destination.write_data_partitioned(partitions=partitions(), overwrite_tables=overwrite_tables)


def push_data_by_copying(source: Path, destination: CloudPath, overwrite_tables: bool) -> None:
    _LOG.info(f"Push data by copying from `{source}` to `{destination}`")
    if destination.exists():
        _LOG.info("Destination location already exists")
        if not overwrite_tables:
            _LOG.error("Dropping tables is disabled")
            raise MostlyDataException(
                "Destination location already exists. If safe, you can enable overwriting data in the destination."
            )
    destination.mkdir(parents=True, exist_ok=True)
    destination.upload_from(source, force_overwrite_to_cloud=True)
    total_size = sum(f.stat().st_size for f in source.rglob("*") if f.is_file()) / (1024 * 1024)
    _LOG.info(f"Finished copying tree of {total_size:.2f} MB")


def adapt_dtypes_to_destination(
    tgt_data: pd.DataFrame,
    schema: Schema,
    tgt: str,
) -> pd.DataFrame:
    tgt_table = schema.tables[tgt]
    pandas_dtypes = {col_name: PandasDType(dtype) for col_name, dtype in dict(tgt_data.dtypes).items()}
    tgt_table_dtypes = tgt_table.dtypes or {}
    tgt_dtypes = pandas_dtypes | tgt_table_dtypes  # use pandas_dtypes as the fallback
    adapted_columns = {}
    for column in tgt_dtypes.keys():
        tgt_column = tgt_data[column]
        column_virtual_dtype = tgt_dtypes[column].to_virtual()
        if pandas_dtypes[column].equivalent(tgt_dtypes[column]):
            # equivalent dtype requires no change
            adapted_columns[column] = tgt_column
        # REFINE below condition to 1. mock data?
        elif column in schema.get_referential_integrity_keys(tgt):  # column must be encompassed
            encompassing_dtype = column_virtual_dtype.encompass(tgt_column)
            if (
                encompassing_dtype.equivalent(pandas_dtypes[column])
                and column in tgt_table_dtypes
                and encompassing_dtype.equivalent(tgt_table_dtypes[column])
            ):
                # equivalent encompassing dtype requires no change
                adapted_columns[column] = tgt_column
            else:  # changing dtype is required
                encompassing_pandas_dtype = PandasDType.from_virtual(encompassing_dtype)
                adapted_columns[column] = tgt_column.astype(encompassing_pandas_dtype.wrapped)
                _LOG.warning(
                    f"Changed dtype from {pandas_dtypes[column]} to {encompassing_pandas_dtype} "
                    f"to encompass column {column}"
                )
                if column in tgt_dtypes:
                    tgt_dtype_cls = type(tgt_dtypes[column])
                    adapted_tgt_dtype = tgt_dtype_cls.from_dtype(encompassing_dtype)
                    _LOG.warning(
                        f"Changing destination dtype of {column=} from {tgt_dtypes[column]} to {adapted_tgt_dtype}"
                    )
                    tgt_table.dtypes[column] = adapted_tgt_dtype
                else:
                    _LOG.warning(f"Cannot change destination dtype of {column=}")
        else:  # coerce
            coerced_column = column_virtual_dtype.coerce(tgt_column)
            adapted_columns[column] = coerced_column
            _LOG.info(f"Coerced column {column}")

    return pd.DataFrame(adapted_columns)
