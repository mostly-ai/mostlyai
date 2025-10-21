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

import concurrent.futures
import hashlib
import json
import logging
import os
import traceback
import uuid
from collections.abc import Callable
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal, Union

import pandas as pd
import sqlparse
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from sqlparse.tokens import DML

from mostlyai.sdk._data.exceptions import MostlyDataException

_ENV_PASSPHRASE = "MOSTLY_PASSPHRASE"

_LOG = logging.getLogger(__name__)


# create a func as_list with one parameter that could be object OR list, use python 3.9 syntax, do not use lambda
def as_list(v: object | list):
    return v if isinstance(v, list) else [v]


def calculate_rows_per_chunk_for_df(df: pd.DataFrame, max_chunk_size_mb: int = 10) -> int:
    """
    Calculate the chunk size based on the dataframe's size.
    Each chunk must have at most `max_chunk_size_mb` megabytes of data.
    """
    if len(df) == 0:
        return 0
    # Calculate the size of the dataframe in bytes
    df_size_bytes = df.memory_usage(deep=True).sum()
    df_size_mb = df_size_bytes / (1024 * 1024)
    # If the dataframe size is less than or equal to `max_chunk_size_mb` MB, take all rows
    if df_size_mb <= max_chunk_size_mb:
        return len(df)
    else:
        # If it's larger, calculate how many rows should be in each
        # chunk so that each chunk is approximately `max_chunk_size_mb` MB
        bytes_per_row = df_size_bytes / len(df)
        return int((max_chunk_size_mb * 1024 * 1024) / bytes_per_row)  # Convert chunk size to megabytes


def create_key(passphrase: str) -> bytes:
    # Create a 32-byte key from the passphrase
    sha3_256 = hashlib.sha3_256()
    sha3_256.update(passphrase.encode("utf-8"))
    key = sha3_256.digest()
    return key


def get_passphrase() -> str:
    if _ENV_PASSPHRASE not in os.environ:
        _LOG.info(f"Environment variable `{_ENV_PASSPHRASE}` is not set. Using default passphrase.")
    passphrase = os.getenv(_ENV_PASSPHRASE, "mostly-secret")
    return passphrase


def encrypt(plain_text: str, passphrase: str) -> str:
    cipher = AES.new(create_key(passphrase), AES.MODE_CBC)
    padded_plain_bytes = pad(plain_text.encode("utf-8"), AES.block_size)
    encrypted_bytes = cipher.iv + cipher.encrypt(padded_plain_bytes)
    return encrypted_bytes.hex()


def decrypt(encrypted_text: str, passphrase: str) -> str:
    try:
        # Decrypt the encrypted text using the passphrase
        encrypted_bytes = bytes.fromhex(encrypted_text)
        iv = encrypted_bytes[:16]
        content = encrypted_bytes[16:]
        cipher = AES.new(create_key(passphrase), AES.MODE_CBC, iv=iv)
        decrypted_padded = cipher.decrypt(content)
        decrypted = unpad(decrypted_padded, AES.block_size)
    except Exception:
        raise MostlyDataException("Cannot decrypt secrets or SSL certificates.")
    return decrypted.decode("utf-8")


def prepare_ssl_path(attribute: str, decryption_path: str) -> str:
    hex_ = uuid.uuid4().hex
    attribute_to_filename = {
        "root_certificate_path": f"ca_{hex_}.crt",
        "ssl_certificate_path": f"client_{hex_}.crt",
        "ssl_certificate_key_path": f"client_{hex_}.key",
    }
    filename = attribute_to_filename.get(attribute, f"default_{hex_}.tmp")
    return str((Path(decryption_path) / filename).absolute())


def validate_gcs_key_file(json_string: str) -> dict:
    """
    Validate the Google Cloud key file.
    If the content is valid, it will return the JSON dict.
    Otherwise, it will return an empty dict.
    """
    try:
        value = json.loads(json_string)
    except (JSONDecodeError, TypeError):
        value = {}
    required_keys = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
    ]
    if not all(key in value for key in required_keys):
        value = {}
    return value


def run_with_timeout_unsafe(f: Callable, *, timeout: float, fallback: Any = None, do_raise: bool = False) -> Any:
    """
    Run a function with a timeout (in seconds).
    Fallback to a value if the function does not return in time.
    The child thread will be killed immediately on timeout (no teardowns).
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    future = executor.submit(f)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        _LOG.info(f"Function `{f.__name__}` did not return in time, falling back to: {fallback}")
        if do_raise:
            raise TimeoutError
        return fallback
    finally:
        executor.shutdown(wait=False)


class absorb_errors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            _LOG.warning(f"Absorbed the following error:\n{traceback.format_exc()}")
        # return True to absorb any exception and not let it propagate
        return True


def strip_column_prefix(
    prefixed_data: pd.Index | list[str] | str,
    table_name: str | None = None,
) -> list[str] | str:
    def infer_table_name(column: str) -> str | None:
        first_token = column.split(TABLE_COLUMN_INFIX)[0]
        if first_token == column:
            raise ValueError("cannot infer table name from column")
        return first_token

    is_str = isinstance(prefixed_data, str)
    if is_str:
        prefixed_data = [prefixed_data]
    if table_name is None:
        table_name = next((infer_table_name(column) for column in prefixed_data), None)
    prefixed_data = [column.removeprefix(table_name + TABLE_COLUMN_INFIX) for column in prefixed_data]
    return prefixed_data[0] if is_str else prefixed_data


def assert_read_only_sql(sql: str) -> None:
    # define allowed read-only operations
    read_only_dml = {"select", "with", "show", "describe", "explain"}

    # define disallowed operations that should be blocked
    disallowed_operations = {
        # data manipulation language (DML)
        "insert",
        "update",
        "delete",
        "merge",
        "replace",
        # data definition language (DDL)
        "create",
        "alter",
        "drop",
        "truncate",
        "rename",
        # data control language (DCL)
        "grant",
        "revoke",
        "deny",
        # transaction control language (TCL)
        "commit",
        "rollback",
        "savepoint",
        # other dangerous operations
        "set",
        "use",
        "exec",
        "execute",
        "call",
        "declare",
        "begin",
        "end",
        "backup",
        "restore",
        "load",
        "import",
        "export",
        "copy",
        "analyze",
        "vacuum",
        "optimize",
        "repair",
        "check",
        "checkpoint",
    }

    # handle empty statements
    if not sql or sql.strip() in {"", ";", "--", "/*", "*/"}:
        raise MostlyDataException("Empty SQL statement.")

    statements = sqlparse.parse(sql)
    if not statements:
        raise MostlyDataException("Empty SQL statement.")

    for statement in statements:
        # check if statement contains only comments or whitespace
        tokens = [t for t in statement.tokens if not t.is_whitespace]
        if not tokens or all(
            t.ttype in (sqlparse.tokens.Comment.Single, sqlparse.tokens.Comment.Multiline) for t in tokens
        ):
            raise MostlyDataException("Empty SQL statement.")

        # get the first token to check for disallowed operations
        first_token = statement.token_first()
        if first_token and first_token.value.lower() in disallowed_operations:
            raise MostlyDataException(f"Disallowed operation: {first_token.value.upper()}")

        # check for DML operations
        for token in statement.flatten():
            if token.ttype is DML:
                dml = token.value.lower()
                if dml not in read_only_dml:
                    raise MostlyDataException(f"Disallowed operation: {dml.upper()}")

        # check for disallowed keywords in proper SQL context
        tokens = [t for t in statement.flatten() if t.ttype is not None]
        in_case = False  # track if we're inside a CASE expression
        for i, token in enumerate(tokens):
            if token.value.lower() == "case":
                in_case = True
            elif token.value.lower() == "end" and in_case:
                in_case = False
                continue  # skip END if it's part of a CASE expression
            elif token.value.lower() in disallowed_operations:
                # check if it's a standalone keyword or part of a larger token
                if i > 0 and tokens[i - 1].value.lower() in {"from", "into", "to", "on", "where", "and", "or", "not"}:
                    continue  # skip if it's part of a larger expression
                raise MostlyDataException(f"Disallowed operation: {token.value.upper()}")


MAX_SCP_SIBLINGS_LIMIT = 50
TABLE_COLUMN_INFIX = "::"
NON_CONTEXT_COLUMN_INFIX = "."
IS_NULL = "_is_null"
SCHEME_SEP = "://"
TEMPORARY_PRIMARY_KEY = "__primary_key"
DATA_TABLE_METADATA_FIELDS = [
    "name",
    "columns",
    "dtypes",
    "primary_key",
    "foreign_keys",
    "encoding_types",
]
Key = str | list[str]
ColumnSort = Union[
    tuple[str, Literal["asc", "desc"]],
    str,
]
OrderBy = ColumnSort | list[ColumnSort]
