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
import logging
import uuid
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd

from mostlyai.sdk._data.base import ForeignKey, NonContextRelation, Schema
from mostlyai.sdk._data.dtype import is_timestamp_dtype
from mostlyai.sdk._data.file.base import LocalFileContainer
from mostlyai.sdk._data.file.table.csv import CsvDataTable
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.non_context import postproc_non_context
from mostlyai.sdk._data.progress_callback import ProgressCallback, ProgressCallbackWrapper
from mostlyai.sdk._data.util.common import (
    IS_NULL,
    NON_CONTEXT_COLUMN_INFIX,
)
from mostlyai.sdk._local.storage import get_model_label
from mostlyai.sdk.domain import Generator, ModelType, SyntheticDataset

_LOG = logging.getLogger(__name__)


def execute_step_finalize_generation(
    *,
    schema: Schema,
    is_probe: bool,
    job_workspace_dir: Path,
    update_progress: ProgressCallback | None = None,
) -> dict[str, int]:
    # get synthetic table usage
    usages = dict()
    for table_name, table in schema.tables.items():
        usages.update({table_name: table.row_count})
    # short circuit for probing
    delivery_dir = job_workspace_dir / "FinalizedSyntheticData"
    if is_probe:
        for table_name in schema.tables:
            finalize_table_generation(
                generated_data_schema=schema,
                target_table_name=table_name,
                delivery_dir=delivery_dir,
                export_csv=False,
            )
        return usages

    random_samples_dir = job_workspace_dir / "RandomSamples"
    zip_dir = job_workspace_dir / "ZIP"

    # calculate total datapoints (rows × columns) across all tables
    total_datapoints = sum(table.row_count * len(table.columns) for table in schema.tables.values())
    export_csv = total_datapoints < 100_000_000  # only export CSV if datapoints < 100M

    with ProgressCallbackWrapper(update_progress, description="Finalize generation") as progress:
        # init progress with total_count; +4 for the 4 steps below
        progress.update(completed=0, total=len(schema.tables) + 4)

        for tgt in schema.tables:
            finalize_table_generation(
                generated_data_schema=schema,
                target_table_name=tgt,
                delivery_dir=delivery_dir,
                export_csv=export_csv,
            )
            progress.update(advance=1)

        _LOG.info("export random samples")
        export_random_samples(
            delivery_dir=delivery_dir,
            random_samples_dir=random_samples_dir,
        )
        progress.update(advance=1)

        _LOG.info("export synthetic data to excel")
        export_data_to_excel(delivery_dir=delivery_dir, output_dir=zip_dir)
        progress.update(advance=1)

        _LOG.info("zip parquet synthetic data")
        zip_data(delivery_dir=delivery_dir, format="parquet", out_dir=zip_dir)
        progress.update(advance=1)

        if export_csv:
            _LOG.info("zip csv synthetic data")
            zip_data(delivery_dir=delivery_dir, format="csv", out_dir=zip_dir)
            progress.update(advance=1)

        return usages


def update_total_rows(synthetic_dataset: SyntheticDataset, usages: dict[str, int]) -> None:
    for table_name, total_rows in usages.items():
        table = next(t for t in synthetic_dataset.tables if t.name == table_name)
        table.total_rows = total_rows


def format_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def postprocess_temp_columns(df: pd.DataFrame, table_name: str, schema: Schema):
    """
    1. remove the suffix of non-context temporary columns `.{parent_table_name}._is_null`
    2. in these columns, replace "True" with a UUID and "False" with pd.NA
    """

    # a variation of mostlyai.sdk._data.non_context.postproc_non_context()
    for relation in schema.relations:
        if not isinstance(relation, NonContextRelation) or relation.child.table != table_name:
            continue
        suffix = NON_CONTEXT_COLUMN_INFIX.join(["", relation.parent.table, IS_NULL])
        temp_columns = [column for column in df.columns if column.endswith(suffix)]
        # fill in some random UUIDs for
        # note: these columns contains strings of boolean values
        for col in temp_columns:
            df[col] = df[col].apply(lambda x: f"mostly{str(uuid.uuid4())[6:]}" if x == "False" else pd.NA)
        # remove suffix
        df.rename(
            columns={c: c.removesuffix(suffix) for c in temp_columns},
            inplace=True,
        )

    return df


def restore_column_order(df: pd.DataFrame, table_name: str, schema: Schema):
    # keep columns in its original order but ignore those which does not exist in the dataframe
    # (e.g., LANGUAGE columns)
    original_columns = [column for column in schema.tables[table_name].columns if column in df.columns]
    df = df[original_columns]
    return df


def export_random_samples_per_table(
    delivery_parquet_dir: Path,
    random_samples_json_path: Path,
    table_name: str,
    schema: Schema | None = None,
    limit: int = 100,
):
    """
    Export random samples of a table from the first parquet file in delivery_parquet_dir
    """
    try:
        # fetch a single parquet file
        parquet_path = next(delivery_parquet_dir.glob("*.parquet"))
    except StopIteration:
        parquet_path = None

    if parquet_path:
        df = pd.read_parquet(parquet_path).sample(frac=1).head(n=limit)
        df = format_datetime(df)
        if schema:
            df = postprocess_temp_columns(df, table_name, schema)
            df = restore_column_order(df, table_name, schema)
        df.to_json(random_samples_json_path, orient="records")
        _LOG.info(f"Export {len(df)} random samples to `{random_samples_json_path}`")


def export_random_samples(
    delivery_dir: Path,
    random_samples_dir: Path,
    limit: int = 100,
):
    """
    Export random samples of all the tables in the delivery directory
    """
    random_samples_dir.mkdir(exist_ok=True, parents=True)
    for path in delivery_dir.glob("*"):
        table_name = path.name
        export_random_samples_per_table(
            delivery_parquet_dir=delivery_dir / table_name / "parquet",
            random_samples_json_path=random_samples_dir / f"{table_name}.json",
            table_name=table_name,
            limit=limit,
        )


def finalize_probing(schema: Schema, delivery_dir: Path):
    for tgt in schema.tables:
        finalize_table_generation(
            generated_data_schema=schema,
            target_table_name=tgt,
            delivery_dir=delivery_dir,
            export_csv=False,
        )


def finalize_table_generation(
    generated_data_schema: Schema,
    target_table_name: str,
    delivery_dir: Path,
    export_csv: bool,
) -> None:
    """
    Post-process the generated data for a given table.
    * handle non-context keys
    * handle reference keys
    * keep only needed columns, and in the right order
    * export to PARQUET, and optionally also to CSV (without col prefixes)
    """

    # read generated "raw" data into memory
    # Note: We should avoid reading all data into memory. E.g. by looping over source.dataset.files or over
    # source.dataset.to_batches(), then process in batches, and then append to CSV, respectively write out
    # as separate parquet files.
    table = generated_data_schema.tables[target_table_name]
    tgt_table_files = table.dataset.files
    n_partitions = len(tgt_table_files)
    _LOG.info(f"POSTPROC will handle {n_partitions} partitions")

    pqt_path = delivery_dir / target_table_name / "parquet"
    pqt_path.mkdir(exist_ok=True, parents=True)
    _LOG.info(f"prepared {pqt_path=} for storing post-processed data as PQT files")

    if export_csv:
        csv_path = delivery_dir / target_table_name / "csv"
        csv_path.mkdir(exist_ok=True, parents=True)
        _LOG.info(f"prepared {csv_path=} for storing post-processed data as CSV file")
    else:
        csv_path = None

    # instantiate container outside of loop to avoid memory to pile up
    container = type(table.container)()
    for part_i, part_file in enumerate(tgt_table_files, start=1):
        container.set_location(part_file)
        part_table = ParquetDataTable(container=container)
        tgt_data = part_table.read_data(do_coerce_dtypes=True)

        # post-process non-context keys
        tgt_data = postproc_non_context(
            tgt_data=tgt_data,
            generated_data_schema=generated_data_schema,
            tgt=target_table_name,
        )

        # keep only original columns, and in the right order
        tgt_cols = (
            cols if (cols := generated_data_schema.tables[target_table_name].columns) is not None else tgt_data.columns
        )
        drop_cols = [c for c in tgt_cols if c not in tgt_data]
        if drop_cols:
            _LOG.info(f"remove columns from final output: {', '.join(drop_cols)}")
        keep_cols = [c for c in tgt_cols if c in tgt_data]
        tgt_data = tgt_data[keep_cols]

        partition_text = f"{part_i} out of {n_partitions}"

        # store post-processed data as PQT files
        _LOG.info(f"store post-processed {partition_text} as PQT")
        pqt_post = ParquetDataTable(path=pqt_path / f"{Path(part_file).stem}.parquet", name=target_table_name)
        pqt_post.write_data(tgt_data)

        # store post-processed data as single CSV file
        if csv_path:
            _LOG.info(f"store post-processed {partition_text} as CSV")
            csv_post = CsvDataTable(path=csv_path / f"{target_table_name}.csv", name=target_table_name)
            csv_post.write_data(tgt_data)

        del tgt_data


def export_data_to_excel(delivery_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # gather data for export
    tables = {}
    is_truncated = False
    is_truncated_note1 = "Note, that this XLSX file only includes a limited number of synthetic samples for each table."
    is_truncated_note2 = "Please refer to alternative download formats, to access the complete generated dataset."
    for table_path in sorted(delivery_dir.glob("*")):
        table_name = table_path.name
        samples = None
        total_row_count = 0
        target_sample_count = 10_000

        for parquet_path in (table_path / "parquet").glob("*.parquet"):
            # fetch a single parquet file at a time
            df = pd.read_parquet(parquet_path)
            total_row_count += df.shape[0]
            if samples is None:
                # attempt to sample from the first partition
                samples = df.head(n=target_sample_count)
                samples = format_datetime(samples)
            else:
                # concatenate samples from different partitions if needed
                remaining_samples = target_sample_count - samples.shape[0]
                if remaining_samples > 0:
                    samples = pd.concat(
                        [samples, df.head(n=remaining_samples)],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    break

        samples = pd.DataFrame() if samples is None else samples

        _LOG.info(f"Exporting table {table_name}: sampling {samples.shape[0]} from {total_row_count} rows")
        tables[table_name] = {
            "data": samples,
            "no_of_included_samples": samples.shape[0],
            "no_of_total_samples": total_row_count,
        }
        if samples.shape[0] < total_row_count:
            is_truncated = True

    # define base format for the excel file
    base_format = {
        "font_color": "#434343",
        "font_name": "Verdana",
        "font_size": 7,
        "valign": "vcenter",
    }

    excel_output_path = Path(output_dir) / "synthetic-samples.xlsx"
    with pd.ExcelWriter(str(excel_output_path), engine="xlsxwriter") as writer:
        workbook = writer.book

        # add formats
        cell_format = workbook.add_format(base_format)
        header_format = workbook.add_format(base_format | {"bold": True, "align": "left"})
        int_format = workbook.add_format(base_format | {"num_format": "#,##0"})
        dt_format = workbook.add_format(base_format | {"num_format": "yyyy-mm-dd"})

        # write a Table of Contents
        toc_sheet_name = "_TOC_"
        worksheet = workbook.add_worksheet(toc_sheet_name)
        worksheet.set_column("A:A", width=40, cell_format=header_format)
        worksheet.set_column("B:C", width=20, cell_format=int_format)
        worksheet.write(0, 0, "Table")
        worksheet.write(0, 1, "No. of Included Samples")
        worksheet.write(0, 2, "No. of Total Samples")
        if is_truncated:
            worksheet.write(len(tables) + 3, 0, is_truncated_note1)
            worksheet.write(len(tables) + 4, 0, is_truncated_note2)
        for idx, (table_name, _) in enumerate(tables.items()):
            worksheet.write_url(
                idx + 1,
                0,
                url=f"internal:'{table_name}'!A1",
                string=table_name,
                cell_format=cell_format,
            )
            worksheet.write(idx + 1, 1, tables[table_name]["no_of_included_samples"])
            worksheet.write(idx + 1, 2, tables[table_name]["no_of_total_samples"])

        # write each DataFrame to a different sheet
        sheet_names_lower = [toc_sheet_name.lower()]
        for table_name, table in tables.items():
            df = table["data"]
            # create a valid sheet name
            sheet_name = table_name[:28]  # consider max sheet name length
            while sheet_name.lower() in sheet_names_lower:
                sheet_name += "_"  # make sheet name unique, with case ignored
            sheet_names_lower.append(sheet_name.lower())
            # add the worksheet
            worksheet = workbook.add_worksheet(sheet_name)
            # set formats, plus adjust the column width for better readability
            for i, column in enumerate(df.columns):
                if is_timestamp_dtype(df[column]):
                    format = dt_format
                else:
                    format = cell_format
                worksheet.set_column(
                    first_col=i,
                    last_col=i,
                    width=12,
                    cell_format=format,
                )
            # set format of header row
            worksheet.set_row(0, height=None, cell_format=header_format)
            # Set the autofilter
            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
            # freeze the first row
            worksheet.freeze_panes(1, 0)
            # write column headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value)
            # write data
            for row_num, row in enumerate(df.values):
                for col_num, _ in enumerate(row):
                    value = df.iloc[row_num, col_num]
                    if not pd.isna(value):
                        worksheet.write(row_num + 1, col_num, df.iloc[row_num, col_num])


def zip_data(delivery_dir: Path, format: Literal["parquet", "csv"], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"synthetic-{format}-data.zip"

    # Choose the compression type based on the file format
    compression = zipfile.ZIP_STORED if format == "parquet" else zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(zip_path, "w", compression) as zipf:
        for table_path in delivery_dir.glob("*"):
            table = table_path.name
            format_path = table_path / format
            for file in format_path.glob("*"):
                zip_loc = f"{table}/{file.relative_to(format_path)}"
                zipf.write(file, arcname=zip_loc)


def create_generation_schema(
    generator: Generator,
    job_workspace_dir: Path,
    step: Literal["pull_context_data", "finalize_generation", "deliver_data"],
) -> Schema:
    tables = {}
    for table in generator.tables:
        # create LocalFileContainer
        container = LocalFileContainer()
        model_label = get_model_label(table, ModelType.tabular, path_safe=True)
        location = str(job_workspace_dir / model_label / "SyntheticData")
        container.set_location(location)
        if step == "pull_context_data":
            columns = None  # HACK: read lazily prefixed columns from parquet file
            is_output = False  # enable lazy reading of properties
        elif step == "finalize_generation":
            columns = [c.name for c in table.columns if c.included]  # use un-prefixed column names
            is_output = False  # enable lazy reading of properties
        elif step == "deliver_data":
            columns = [c.name for c in table.columns if c.included]  # use un-prefixed column names
            is_output = True
        else:
            raise ValueError(f"Unsupported step: {step}")
        # create ParquetDataTable
        data_table = ParquetDataTable(container=container)
        data_table.name = table.name
        data_table.primary_key = table.primary_key
        data_table.columns = columns
        data_table.encoding_types = {c.name: c.model_encoding_type for c in table.columns if c.included}
        data_table.is_output = is_output
        data_table.foreign_keys = [
            ForeignKey(column=fk.column, referenced_table=fk.referenced_table, is_context=fk.is_context)
            for fk in table.foreign_keys or []
        ]
        tables[table.name] = data_table
    return Schema(tables=tables)
