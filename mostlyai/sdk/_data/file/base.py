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

import abc
import functools
import logging
import re
import time
from abc import abstractmethod
from collections.abc import Generator, Iterable
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import duckdb
import pandas as pd
import pyarrow
import pyarrow as pa
import pyarrow.dataset as ds
from cloudpathlib import AnyPath
from fsspec.implementations.local import LocalFileSystem
from pyarrow import ArrowIndexError, ArrowInvalid

from mostlyai.sdk._data.base import (
    DataContainer,
    DataTable,
    order_df_by,
)
from mostlyai.sdk._data.dtype import (
    PandasDType,
    coerce_dtypes_by_encoding,
    pyarrow_to_pandas_map,
)
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import DATA_TABLE_METADATA_FIELDS, SCHEME_SEP, OrderBy, assert_read_only_sql

FILE_DATA_TABLE_LAZY_INIT_FIELDS = DATA_TABLE_METADATA_FIELDS + [
    "columns",
    "dataset",
]

_LOG = logging.getLogger(__name__)


class FileType(Enum):
    """
    File types
    """

    csv = "CSV"
    tsv = "TSV"
    parquet = "PARQUET"
    feather = "FEATHER"
    json = "JSON"
    compressed = "COMPRESSED"
    unknown = "UNKNOWN"
    folder = "FOLDER"


VALID_FILE_TYPES = [
    FileType.parquet,
    FileType.feather,
    FileType.csv,
    FileType.tsv,
    FileType.json,
]
FILE_TYPE_COMPRESSED = ["gz", "bz2"]
EXT_TO_FILE_TYPE = {
    "csv": FileType.csv,
    "tsv": FileType.tsv,
    "parquet": FileType.parquet,
    "pqt": FileType.parquet,
    "feather": FileType.feather,
    "json": FileType.json,
    "jsonl": FileType.json,
}


def get_file_name_and_type(full_file_name: str | AnyPath) -> tuple[str, FileType]:
    if isinstance(full_file_name, str):
        full_file_name = Path(full_file_name)
    file_name = full_file_name.stem
    if len(file_list := full_file_name.suffix.split(".")) == 1:
        return file_name, FileType.folder
    ext = file_list[1]
    if ext in FILE_TYPE_COMPRESSED:
        ext = Path(file_name).suffix.split(".")[1]
        file_name = Path(file_name).stem
    file_type = EXT_TO_FILE_TYPE.get(ext, FileType.unknown)
    return file_name, file_type


class FileDataTable(DataTable, abc.ABC):
    LAZY_INIT_FIELDS = frozenset(FILE_DATA_TABLE_LAZY_INIT_FIELDS)
    IS_WRITE_APPEND_ALLOWED: bool = False  # specified for each format individually

    def __init__(
        self,
        *args,
        path: Any | None = None,
        **kwargs,
    ):
        if "container" not in kwargs:
            kwargs["container"] = self.container_class()(file_path=path)
        super().__init__(*args, **kwargs)
        self.path: str | Path = path or self.container.path
        if self.path and isinstance(self.path, (str, Path)):
            self.name = self.name or get_file_name_and_type(Path(self.path))[0]
        self.dataset: ds.Dataset | None = None
        self.local_cache_path: str | Path | None = None

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, is_output={self.is_output}, path={self.path})"

    @classmethod
    def from_file_path(cls, path: str | Path, **kwargs) -> "FileDataTable":
        file_path = Path(urlparse(str(path)).path)
        return cls(path=file_path, **kwargs)

    @classmethod
    def dtype_class(cls) -> type:
        return PandasDType

    def fetch_dtypes(self) -> dict[str, Any]:
        try:
            dtypes = {}
            for field in self.dataset.schema:
                if isinstance(field.type, pa.DictionaryType):
                    dtypes[field.name] = pd.CategoricalDtype()
                else:
                    dtypes[field.name] = pyarrow_to_pandas_map.get(field.type)
        except ArrowIndexError:
            # handle case when no records exist
            dtypes = {c.name: pd.ArrowDtype(pa.string()) for c in self.dataset.schema}
        except ArrowInvalid as err:
            # handle case when data cannot be read
            _LOG.error(f"{err=} of {type(err)=}")
            raise MostlyDataException("Error while reading data.")
        return dtypes

    def _get_dtypes(self):
        assert not self.is_output
        dtypes = self.fetch_dtypes()
        return {c: (self.dtype_class())(wrapped=dtypes[c]) for c in self.columns}

    def _lazy_fetch(self, item: str) -> None:
        if item == "name":
            self.name = self.container.name if self.container and self.container.name is not None else ""
        elif item == "dtypes":
            self.dtypes = self._get_dtypes()
        elif item == "dataset":
            self.dataset = self._get_dataset()
        elif item == "columns":
            self.columns = self._get_columns()
        else:
            super()._lazy_fetch(item)

    def is_accessible(self) -> bool:
        if self.path:
            path = self.path if isinstance(self.path, Path) else Path(self.path)
            return path.exists()

    def drop(self, drop_all: bool = False):
        if self.is_accessible():
            if self.path.is_file():
                self.path.unlink()
            elif self.path.is_dir():
                self.path.rmdir()

    def _build_ds_filter(self, where: dict[str, Any] | None = None):
        def are_types_compatible(data_type: pa.DataType, vals: list) -> bool:
            type_mapping = {
                pa.types.is_string: str,
                pa.types.is_integer: int,
                pa.types.is_floating: float,
                pa.types.is_boolean: bool,
                pa.types.is_null: type(None),
            }

            if len(vals) == 0 and pa.types.is_null(data_type):
                return True

            for arrow_check, python_type in type_mapping.items():
                if arrow_check(data_type):
                    return all(isinstance(val, (python_type, type(None))) for val in vals)

            return False  # unsupported type

        filter = None
        if where:
            filters = []
            for c, v in where.items():
                field_type = self.dataset.schema.field(c).type
                # make sure values is a list of unique values
                values = list(set(v)) if (isinstance(v, Iterable) and not isinstance(v, str)) else [v]
                # sanitize values before creating a filter
                if any(pd.isna(v) for v in values):
                    # map null types to python's None
                    values = [v for v in values if not pd.isna(v)] + [None]
                if field_type == pa.null():
                    # handle empty or null-only columns
                    values = [None] if None in values else []
                if are_types_compatible(field_type, values):
                    field_c = ds.field(c)
                else:
                    values = [str(val) if val is not None else None for val in values]
                    field_c = ds.field(c).cast(pa.string())
                filters.append(field_c.isin(values))
            filter = functools.reduce(lambda x, y: x & y, filters)
        return filter

    def read_chunks(
        self,
        where: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        do_coerce_dtypes: bool = True,
        fetch_chunk_size: int | None = None,
        yield_chunk_size: int | None = None,
    ):
        t00 = time.time()
        fetch_chunk_size = fetch_chunk_size if fetch_chunk_size is not None else 1_000_000
        yield_chunk_size = yield_chunk_size if yield_chunk_size is not None else fetch_chunk_size
        filter = self._build_ds_filter(where)
        iterator = self.dataset.scanner(
            columns=columns,
            filter=filter,
            batch_size=fetch_chunk_size,
        ).to_batches()
        total_time = 0
        chunks = []
        chunk_idx = 0

        def yield_data():
            nonlocal chunks
            chunk_df = pa.Table.from_batches(chunks).to_pandas(
                # convert to pyarrow DTypes
                types_mapper=pyarrow_to_pandas_map.get,
                # reduce memory of conversion
                # see https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
                split_blocks=True,
                self_destruct=True,
            )
            if do_coerce_dtypes:
                chunk_df = coerce_dtypes_by_encoding(chunk_df, self.encoding_types)
            chunks = []
            # return a copy of the chunk to avoid memory leak
            chunk_df = chunk_df.copy()
            yield chunk_df

        while True:
            t0 = time.time()
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            # accumulate chunks
            chunks.append(chunk)
            if sum(len(c) for c in chunks) >= yield_chunk_size:
                # yield accumulated data once it reaches yield_chunk_size
                yield from yield_data()
            chunk_idx += 1
            total_time += time.time() - t0
            if chunk_idx % 10 == 0:
                _LOG.info(f"processed {chunk_idx} chunks in {total_time:.2f}s")
        if len(chunks) > 0:
            # yield remaining data
            yield from yield_data()
        _LOG.info(f"finished reading {chunk_idx} chunks in {time.time() - t00:.2f}s")

    def read_data(
        self,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        shuffle: bool | None = False,
        order_by: OrderBy | None = None,
        do_coerce_dtypes: bool | None = False,
    ) -> pd.DataFrame:
        t0 = time.time()
        filter = self._build_ds_filter(where)
        df = (
            self.dataset.scanner(columns=columns, filter=filter)
            .to_table()
            .to_pandas(
                # convert to pyarrow DTypes
                types_mapper=pyarrow_to_pandas_map.get,
                # reduce memory of conversion
                # see https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
                split_blocks=True,
            )
        )
        if shuffle:
            df = df.sample(frac=1)
        if limit is not None:
            df = df.head(limit)
        if order_by:
            df = order_df_by(df, order_by)
        if do_coerce_dtypes:
            df = coerce_dtypes_by_encoding(df, self.encoding_types)
        df = df.reset_index(drop=True)
        _LOG.info(f"read {self.DATA_TABLE_TYPE} data `{self.name}` {df.shape} in {time.time() - t0:.2f}s")
        return df

    def write_data_partitioned(
        self,
        partitions: Generator[tuple[str, pd.DataFrame], None, None],
        **kwargs,
    ) -> None:
        if self.IS_WRITE_APPEND_ALLOWED:
            # for the case of "write append allowed" reuse the default implementation to write to a single file
            return super().write_data_partitioned(partitions)
        else:
            # for the case of "write append disallowed", write each partition to a designated file
            # ensure base directory (where partitioned files will be stored) exists
            self.container.path.mkdir(exist_ok=True, parents=True)
            # make a copy of the original container, to modify the uri for each part
            container = type(self.container)(**self.container.__dict__)
            for part_file, data in partitions:
                part_path = self.container.path / f"{Path(part_file).stem}.{self.DATA_TABLE_TYPE}"
                container.set_location(str(part_path))
                part_table_output = type(self)(container=container)
                part_table_output.write_data(df=data)

    @functools.cached_property
    def row_count(self) -> int:
        # reading out num_rows per batch was measured to be ~50% faster than dataset.scanner().count_rows()
        return sum(batch.num_rows for batch in self.dataset.scanner(columns=[]).to_batches())

    def _get_dataset(self, *args, **kwargs) -> ds.Dataset:
        try:
            d = ds.dataset(
                source=self.container.valid_files_without_scheme,
                filesystem=self.container.file_system,
                format=self._get_dataset_format(*args, **kwargs),
                exclude_invalid_files=False,
            )
        except pyarrow.ArrowInvalid:
            message = "There was an issue with the uploaded file. Please verify that it's a valid format."
            _LOG.error(message)
            raise MostlyDataException(message)
        except FileNotFoundError:
            message = "File not found."
            _LOG.error(message)
            raise MostlyDataException(message)
        except Exception:
            message = "Encountered an unknown error when reading the file."
            _LOG.error(message)
            raise MostlyDataException(message)
        return d

    def _get_dataset_format(self) -> ds.FileFormat:
        pass

    def _get_columns(self) -> list[str]:
        return [c.name for c in self.dataset.schema if c.name != ""]

    def handle_if_exists(self, if_exists: str = "fail") -> str:
        destination_exists = self.container.path.exists()
        if destination_exists:
            if if_exists == "fail":
                raise MostlyDataException("Destination location already exists.")
            elif if_exists == "append" and self.IS_WRITE_APPEND_ALLOWED:
                return "a"
        return "w"


class FileContainer(DataContainer):
    SCHEMES = ["http", "https"]
    DEFAULT_SCHEME = ""
    DELIMITER_SCHEMA = ""

    @property
    def path_prefix(self):
        return f"{self.DEFAULT_SCHEME}{SCHEME_SEP}"  # e.g. file://

    @property
    def delimiter_prefix(self):
        return f"{self.DELIMITER_SCHEMA}{SCHEME_SEP}"  # e.g. file://

    @property
    @abc.abstractmethod
    def path(self) -> AnyPath:
        pass

    @classmethod
    def _re_match_uri(cls, uri: str) -> re.Match:
        prefix = f"{cls.DEFAULT_SCHEME}{SCHEME_SEP}"
        if "//" not in uri:
            uri = f"{prefix}{uri}"
        match = re.match(rf"{prefix}([^/]+)/(.*)", uri)
        return match

    @abc.abstractmethod
    def set_uri(self, uri: str): ...

    @property
    def path_str(self) -> str:
        return str(self.path)

    @property
    def path_without_scheme(self) -> str:
        scheme_loc = self.path_str.find(SCHEME_SEP)
        cutoff = scheme_loc + len(SCHEME_SEP) if scheme_loc > 0 else 0
        return self.path_str[cutoff:]

    @property
    def valid_files_without_scheme(self) -> list[str]:
        scheme_loc = self.path_str.find(SCHEME_SEP)
        cutoff = scheme_loc + len(SCHEME_SEP) if scheme_loc > 0 else 0
        valid_files = self.list_valid_files()
        return [str(file)[cutoff:] for file in valid_files]

    @property
    def uri(self) -> str:
        return self.path.as_uri()

    @property
    def name(self):
        return get_file_name_and_type(self.path_str)[0]

    @property
    def storage_options(self) -> dict | None:
        """
        Pandas' storage_options: Extra options that make sense for a particular storage connection.
        For example, applicable for cloud storage, where authentication is required.
        :return: dict of those extra options, if applicable. Otherwise, None
        """
        return None

    @property
    def transport_params(self) -> dict | None:
        """
        transport_params are optionally required by smart_open, e.g. in cases where authentication is needed.
        :return: dict of those transport_params, if applicable. Otherwise, None
        """
        return None

    @property
    @abstractmethod
    def file_system(self) -> Any:
        pass

    def ls(self) -> list[AnyPath]:
        if not self.path or not self.path.exists():
            return []
        elif self.path.is_dir():
            return sorted(list(self.path.glob("*")))
        else:
            return [self.path]

    def list_valid_files(self) -> list[AnyPath]:
        valid_files = {}
        for file in self.ls():
            _, file_type = get_file_name_and_type(str(file))
            if file_type in VALID_FILE_TYPES:
                valid_files[file_type] = valid_files.get(file_type, []) + [file]
        # we use order of VALID_FILE_TYPES to give precedence in case of mixed-type folders

        for file_type in VALID_FILE_TYPES:
            if file_type in valid_files:
                return valid_files[file_type]
        return []

    def is_accessible(self) -> bool:
        return self.path.exists()

    def directories(self) -> list[str]:
        """
        Directory names in the given path.
        :return: list of directories' names within self.path (1st level)
        """
        return [str(p) for p in self.path.iterdir() if p.is_dir()]

    def files(self) -> list[str]:
        """
        File names in the given path.
        :return: list of files' names within self.path (1st level)
        """
        return [str(p) for p in self.path.iterdir() if not p.is_dir()]

    def set_location(self, location: str) -> dict:
        self.set_uri(location)
        return {"location": location}

    def _init_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        pass

    @staticmethod
    def _create_duckdb_secret(con: duckdb.DuckDBPyConnection, secret_params: dict[str, Any]) -> None:
        def sql_escape(val: str) -> str:
            return str(val).replace("'", "''")  # escape single quotes for SQL strings

        # all values must be wrapped in single quotes and escaped
        parts = [f"{key} '{sql_escape(value)}'" for key, value in secret_params.items()]
        joined_parts = ",\n    ".join(parts)
        create_secret_sql = f"CREATE SECRET (\n    {joined_parts}\n);"

        con.execute(create_secret_sql)

    def query(self, sql: str) -> pd.DataFrame:
        assert_read_only_sql(sql)
        try:
            with duckdb.connect(database=":memory:") as con:
                con.execute("""
                    SET disabled_filesystems = 'LocalFileSystem';
                    SET allow_community_extensions = false;
                    SET lock_configuration = true;
                """)
                self._init_duckdb(con)
                result = con.execute(sql).fetchdf()

            return result
        except duckdb.IOException as e:
            _LOG.error(f"IO Error executing query: {str(e)}")
            raise
        except duckdb.Error as e:
            _LOG.error(f"DuckDB Error executing query: {str(e)}")
            raise


class LocalFileContainer(FileContainer):
    SCHEMES = ["file"]
    DEFAULT_SCHEME = "file"
    DELIMITER_SCHEMA = "file"

    def _sanitized_path(self, path):
        if isinstance(path, str) and self.path_prefix in path:
            return path.split(SCHEME_SEP)[1]
        elif isinstance(path, (str, Path)):
            return path
        else:
            return Path("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = Path(self._sanitized_path(kwargs.get("file_path")))
        self.fs = LocalFileSystem()
        if not self._path:
            raise MostlyDataException("Provide the file path.")

    def set_uri(self, uri: str):
        self._path = Path(uri)

    def drop_all(self):
        pass

    @property
    def path(self) -> AnyPath:
        return self._path

    @property
    def file_system(self) -> Any:
        return self.fs

    def is_accessible(self) -> bool:
        if not self.file_system.exists(self.path):
            raise MostlyDataException("Cannot access file.")
        return True
