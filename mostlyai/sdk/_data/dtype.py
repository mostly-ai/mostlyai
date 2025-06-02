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
DType, along with its subclasses and helper functions, provide
the necessary base to handle data types in a heterogeneous environment.
"""

import abc
import logging
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import pyarrow as pa

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import absorb_errors
from mostlyai.sdk.domain import ModelEncodingType

STRING = "string[pyarrow]"  # This utilizes pyarrow's large string type since pandas 2.2
INT64 = "int64[pyarrow]"
FLOAT64 = "float64[pyarrow]"
BOOL = "bool[pyarrow]"


_LOG = logging.getLogger(__name__)


class DType(abc.ABC):
    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    @classmethod
    @abc.abstractmethod
    def from_dtypes(cls, dtypes: dict[str, "DType"]) -> dict[str, "DType"]:
        """
        Create a dictionary of {column: DType} from a given one to correspond to the given class
        :param dtypes: input dtypes to convert to dtypes of the given class
        """
        ...

    def equivalent(self, other):
        """
        Checks whether self is equal or of a comparable data type with the other object
        :param other: other object
        :return: True if equivalent, False otherwise
        """
        if type(self) is type(other):
            return self.__dict__ == other.__dict__
        elif isinstance(other, DType):
            return self.to_virtual() == other.to_virtual()
        else:
            return False

    @abc.abstractmethod
    def to_virtual(self) -> "VirtualDType":
        """
        Convert any DType to a VirtualDType

        :return: VirtualDType
        """
        ...

    def coerce(self, data: pd.Series) -> pd.Series:
        """
        Coerce a given the data to self's data type (expected to be overriden for each specific coercion)
        :param data: data
        :return: coerced pd.Series of data
        """
        return data  # TODO consider bringing to this level or make abstract

    def cast(self, data: pd.Series) -> pd.Series:
        """
        Lossless cast of data, if possible, to this DType. Otherwise, the data itself

        :return: pd.Series
        """
        return data.convert_dtypes()

    def encompasses(self, data: pd.Series) -> bool:
        """
        Answers whether self's data type encompasses all records of data
        :param data: data
        :return: True/False
        """
        try:
            lossless_data = self.cast(data)
            coerced_data = self.coerce(data)
            lossless_null_map = lossless_data.isnull()
            coerced_null_map = coerced_data.isnull()
            return lossless_null_map.equals(coerced_null_map) and bool(
                all(lossless_data[~lossless_null_map] == coerced_data[~lossless_null_map])
            )
        except Exception as e:
            _LOG.warning(f"encompasses failed for {type(self)=} and {data.dtype=}: {e}")
            return False  # in case of a failure, be on the safer side

    def get_encompass_data_fallback_dtypes(self) -> list["DType"]:
        """Returns a default list of fallback dtypes for data encompassment (shall be overriden, when needed)"""
        return [VirtualVarchar]

    def encompass_data(
        self,
        data: pd.Series,
        fallback_dtypes: list["DType"] = None,
        is_last_fallback_always_valid: bool = True,
    ) -> "DType":
        """
        A common procedure to encompass any data by given steps: fallback data types.
        :param data: data to encompass
        :param fallback_dtypes: ordered fallback strategy
        :param is_last_fallback_always_valid: whether fallback_dtypes[-1] is valid as the last fallback
        :return: the first dtype in [self] + fallback_dtypes that encompasses data or None
        """
        if not fallback_dtypes:
            fallback_dtypes = self.get_encompass_data_fallback_dtypes()
        if self.encompasses(data):
            return self

        valid_fallback = fallback_dtypes[-1] if is_last_fallback_always_valid else None
        for fallback_dtype in fallback_dtypes:
            if fallback_dtype == valid_fallback or fallback_dtype().encompasses(data):
                encompassing_dtype = fallback_dtype
                break

        _LOG.warning(f"Changing {self} to {encompassing_dtype} to encompass data {data.name} of length {len(data)}")
        return encompassing_dtype()

    def encompass(self, data: pd.Series) -> "DType":
        """
        Encompass any data to providing a (potentially minimal) matching DType for it

        :return: DType
        """
        return self.encompass_data(data)


class VirtualDType(DType, abc.ABC):
    def __repr__(self):
        return f"{type(self).__name__}()"

    @classmethod
    def from_dtypes(cls, dtypes: dict[str, DType]) -> dict[str, "VirtualDType"]:
        return {c: t.to_virtual() for c, t in dtypes.items()}

    def to_virtual(self) -> "VirtualDType":
        return self


class VirtualVarchar(VirtualDType):
    def __init__(self, length: int = None):
        self.length = length

    def coerce(self, data: pd.Series) -> pd.Series:
        return str_coerce(data, self.length)

    def cast(self, data: pd.Series) -> pd.Series:
        return data.astype(STRING)

    def encompass(self, data: pd.Series) -> "DType":
        if self.encompasses(data):
            return self
        max_length = data.astype(STRING).fillna("").apply(len).max()
        _LOG.warning(
            f"Changing {self} to VirtualVarchar with length={max_length} to encompass data of length{len(data)}"
        )
        return type(self)(length=max_length)


class VirtualBoolean(VirtualDType):
    def coerce(self, data: pd.Series) -> pd.Series:
        return bool_coerce(data)


class VirtualInteger(VirtualDType):
    def get_encompass_data_fallback_dtypes(self) -> list["DType"]:
        return [VirtualFloat, VirtualVarchar]

    def coerce(self, data: pd.Series) -> pd.Series:
        return int_coerce(data)

    def cast(self, data: pd.Series) -> pd.Series:
        return pd.to_numeric(data, errors="ignore", downcast="integer", dtype_backend="pyarrow")


class VirtualFloat(VirtualDType):
    def coerce(self, data: pd.Series) -> pd.Series:
        return float_coerce(data)

    def cast(self, data: pd.Series) -> pd.Series:
        return pd.to_numeric(data, errors="ignore", downcast="float", dtype_backend="pyarrow")


class VirtualDate(VirtualDType):
    def coerce(self, data: pd.Series) -> pd.Series:
        return datetime_coerce(data)


class VirtualDatetime(VirtualDType):
    def coerce(self, data: pd.Series) -> pd.Series:
        return datetime_coerce(data)


class VirtualTimestamp(VirtualDType):
    def coerce(self, data: pd.Series) -> pd.Series:
        return datetime_coerce(data)


class WrappedDType(DType):
    def __init__(self, wrapped: Any):
        self.wrapped = wrapped

    def __repr__(self):
        return f"{type(self).__name__}(wrapped={self.wrapped})"

    def __eq__(self, other):
        return type(self) is type(other) and str(self.wrapped) == str(other.wrapped)

    @classmethod
    @abc.abstractmethod
    def from_virtual(cls, dtype: VirtualDType) -> "WrappedDType": ...

    @classmethod
    def from_dtype(cls, dtype: DType) -> "WrappedDType":
        if isinstance(dtype, cls):
            return dtype
        elif isinstance(dtype, WrappedDType):
            return cls.from_virtual(dtype.to_virtual())
        elif isinstance(dtype, VirtualDType):
            return cls.from_virtual(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    @classmethod
    def from_dtypes(cls, dtypes: dict[str, DType]) -> dict[str, "WrappedDType"]:
        wrapped_dtypes = {}
        for column, dtype in dtypes.items():
            wrapped_dtypes[column] = cls.from_dtype(dtype)
        return wrapped_dtypes

    @classmethod
    def from_dtypes_unwrap(cls, dtypes: dict[str, DType]) -> dict[str, Any]:
        if not dtypes:
            return {}
        wrapped_dtypes = cls.from_dtypes(dtypes)
        unwrapped_dtypes = {c: t.wrapped for c, t in wrapped_dtypes.items()}
        return unwrapped_dtypes

    @classmethod
    @abc.abstractmethod
    def parse(cls, dtype: str) -> Optional["WrappedDType"]: ...


def coerce_dtype_by_encoding(
    x: pd.Series,
    encoding_type: ModelEncodingType | None = None,
) -> pd.Series:
    """Coerce dtype based on ModelEncodingType.
    Coerce values to a specific dtype, based on the specified ModelEncodingType. Any values that cannot be coerced will be
    replaced by missing values.
    """
    if str(x.dtype).lower().startswith("binary"):
        # handle case where some non-UTF8 chars result in binary dtype for whole column
        x = x.astype("object")
        x = x.str.decode("UTF-8", "backslashreplace")
        x = x.astype(STRING)

    if encoding_type in [
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_binned,
        ModelEncodingType.tabular_numeric_digit,
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.language_numeric,
    ]:
        if pd.api.types.is_bool_dtype(x):
            # convert booleans to integer -> True=1, False=0
            x = x.astype("Int8")
        elif not pd.api.types.is_numeric_dtype(x):
            x = x.astype(str)
            # remove any currency symbols
            x = x.str.replace("[$€£¥]", "", regex=True)
            # convert other non-numerics to string, and extract valid numeric sub-string
            valid_num = r"(-?[0-9]*[\.]?[0-9]+(?:[eE][+\-]?\d+)?)"
            x = x.str.extract(valid_num, expand=False)
            # convert, and coerce any errors to NAs
            x = pd.to_numeric(x, errors="coerce")
        # convert to numpy_nullable due to https://github.com/apache/arrow/issues/35273
        if pd.api.types.is_integer_dtype(x):
            x = x.astype("Int64")
        else:
            x = x.astype("Float64")
    elif encoding_type in [
        ModelEncodingType.tabular_datetime,
        ModelEncodingType.tabular_datetime_relative,
        ModelEncodingType.language_datetime,
    ]:
        # convert all others to pyarrow timestamp; and coerce any errors to NAs
        x = pd.to_datetime(x, errors="coerce", utc=True)
        # currently we do not retain timezone info
        x = x.dt.tz_localize(None)
        # convert to timestamp with ns resolution
        x = x.astype("datetime64[ns]")
    elif encoding_type in [
        ModelEncodingType.tabular_categorical,
        ModelEncodingType.tabular_lat_long,
        ModelEncodingType.tabular_character,
        ModelEncodingType.language_text,
        ModelEncodingType.language_categorical,
    ]:
        x = x.astype(STRING)
    elif encoding_type is None or encoding_type == ModelEncodingType.auto:
        # treat keys as strings
        if pd.api.types.is_float_dtype(x):
            # convert float values that represent whole numbers to integers
            if x.dropna().apply(lambda v: float(v).is_integer()).all():
                x = x.astype("Int64")
        x = x.astype(STRING)
    else:
        raise MostlyDataException(f"Incorrect ModelEncodingType {encoding_type} for {x.name}.")
    return x


def coerce_dtypes_by_encoding(
    df: pd.DataFrame,
    encoding_types: dict[str, ModelEncodingType] | None = None,
) -> pd.DataFrame:
    """Convert dtypes based on EncodingTypes."""
    if len(df.columns) == 0:
        return df
    encoding_types = encoding_types or {}
    return pd.concat(
        [coerce_dtype_by_encoding(df[col], encoding_types.get(col)) for col in df.columns],
        axis=1,
    )


V_DTYPE_ENCODING_TYPE_MAP = defaultdict(
    lambda: ModelEncodingType.tabular_categorical,
    {
        VirtualVarchar: ModelEncodingType.tabular_categorical,
        VirtualBoolean: ModelEncodingType.tabular_categorical,
        VirtualInteger: ModelEncodingType.tabular_numeric_auto,
        VirtualFloat: ModelEncodingType.tabular_numeric_auto,
        VirtualDate: ModelEncodingType.tabular_datetime,
        VirtualDatetime: ModelEncodingType.tabular_datetime,
        VirtualTimestamp: ModelEncodingType.tabular_datetime,
    },
)


# COERCION FUNCTIONS #


def bool_coerce(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        # exit early if series type is boolean
        return s.astype(BOOL)
    # count nulls before coercion
    n_na_0 = sum(pd.isna(s))
    # replace True -> 1, False -> 0
    s = s.astype(object).replace({"True": 1, "False": 0})
    # coerce to numeric
    s = pd.to_numeric(s, errors="coerce")
    # replace all values that are not equivalent to {1, 0} with pd.NA
    s[~s.isin([0, 1])] = pd.NA
    # count nulls after coercion
    n_na_1 = sum(pd.isna(s))
    # report coercion
    n_coerced = n_na_1 - n_na_0
    if n_coerced > 0:
        _LOG.warning(f"{n_coerced} values coerced during bool_coerce")
    # map to BOOL
    return s.astype(BOOL)


def datetime_coerce(s: pd.Series) -> pd.Series:
    # count nulls before coercion
    n_na_0 = sum(pd.isna(s))
    # coerce to datetime
    s = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    # count nulls after coercion
    n_na_1 = sum(pd.isna(s))
    # report coercion
    n_coerced = n_na_1 - n_na_0
    if n_coerced > 0:
        _LOG.warning(f"{n_coerced} values coerced during datetime_coerce")
    # map to "datetime64[ns]"
    return s.astype("datetime64[ns]")


def float_coerce(s: pd.Series) -> pd.Series:
    # count nulls before coercion
    n_na_0 = sum(pd.isna(s))
    # coerce to float
    s = pd.to_numeric(s, errors="coerce", downcast="float")
    # count nulls after coercion
    n_na_1 = sum(pd.isna(s))
    # report coercion
    n_coerced = n_na_1 - n_na_0
    if n_coerced > 0:
        _LOG.warning(f"{n_coerced} values coerced during float_coerce")
    return s.astype(FLOAT64)


def int_coerce(s: pd.Series) -> pd.Series:
    # count nulls before coercion
    n_na_0 = sum(pd.isna(s))
    # coerce to integer
    s = pd.to_numeric(s, errors="coerce", downcast="integer")
    # count nulls after coercion
    n_na_1 = sum(pd.isna(s))
    # convert to "Int64" to handle Arrow's unsupported NaNs
    s_floor = s.floordiv(1).astype("Int64")
    # absorb any errors, if occurred during coercion stats logging
    with absorb_errors():
        # report coercion
        n_coerced = n_na_1 - n_na_0
        if n_coerced > 0:
            _LOG.warning(f"{n_coerced} values null-coerced during int_coerce")
        n_coerced = sum(s[s.notna()] != s_floor[s.notna()])
        if n_coerced > 0:
            _LOG.warning(f"{n_coerced} values downcast with loss during int_coerce")
    # return floor series of type INT64 (pyarrow)
    return s_floor.astype(INT64)


def str_coerce(s: pd.Series, max_length: int | None = None) -> pd.Series:
    # only bounded string-like SQL dtypes have length attribute
    # e.g. VARCHAR(255), but not TEXT()
    # map to STRING dtype
    s = s.astype(STRING)
    if max_length is not None:
        # clip to max length
        s_trimmed = s.str.slice(0, max_length)
        # report coercion
        n_coerced = sum(s[s.notna()] != s_trimmed[s.notna()])
        if n_coerced > 0:
            _LOG.warning(f"{n_coerced} values trimmed to max length of {max_length} during str_coerce")
        # return trimmed series
        s = s_trimmed
    return s


def time_coerce(s: pd.Series) -> pd.Series:
    # count nulls before coercion
    n_na_0 = sum(pd.isna(s))
    # represent as string
    s = s.astype(STRING)
    # coerce to datetime
    s = pd.to_datetime(s, errors="coerce")
    # take out time component
    s = s.dt.time
    # count nulls after coercion
    n_na_1 = sum(pd.isna(s))
    # report coercion
    n_coerced = n_na_1 - n_na_0
    if n_coerced > 0:
        _LOG.warning(f"{n_coerced} values coerced during datetime_coerce")
    # map to "object"
    return s.astype("object")


class PandasDType(WrappedDType):
    @classmethod
    def from_virtual(cls, dtype: VirtualDType) -> "WrappedDType":
        if isinstance(dtype, VirtualBoolean):
            return cls(BOOL)
        elif isinstance(dtype, VirtualInteger):
            return cls(INT64)
        elif isinstance(dtype, VirtualFloat):
            return cls(FLOAT64)
        elif isinstance(dtype, VirtualVarchar):
            return cls(STRING)
        elif isinstance(dtype, VirtualDatetime):
            return cls("datetime64[ns]")
        elif isinstance(dtype, VirtualTimestamp):
            return cls("datetime64[ns]")
        else:
            raise NotImplementedError(f"handler for virtual {dtype=} is not specified")

    def to_virtual(self) -> VirtualDType:
        dtype = str(self.wrapped)
        if dtype.startswith("bool"):
            return VirtualBoolean()
        elif dtype.lower().startswith("int"):
            return VirtualInteger()
        elif dtype.lower().startswith("float") or dtype.startswith("double"):
            return VirtualFloat()
        elif dtype.startswith("str"):
            return VirtualVarchar()
        elif dtype.startswith("binary"):
            return VirtualVarchar()
        elif dtype == "object":
            return VirtualVarchar()
        elif dtype == "category":
            return VirtualVarchar()
        elif dtype.startswith("null"):
            return VirtualVarchar()
        elif dtype.startswith("date") or dtype.startswith("time"):
            return VirtualTimestamp()
        else:
            _LOG.warning(f"no VirtualDtype found for {dtype}")
            return VirtualVarchar()

    @classmethod
    def parse(cls, dtype: str) -> Optional["WrappedDType"]:
        return cls(dtype)


pyarrow_to_pandas_map = {
    pa.null(): pd.ArrowDtype(pa.null()),
    pa.bool_(): pd.ArrowDtype(pa.bool_()),
    pa.int8(): pd.ArrowDtype(pa.int8()),
    pa.int16(): pd.ArrowDtype(pa.int16()),
    pa.int32(): pd.ArrowDtype(pa.int32()),
    pa.int64(): pd.ArrowDtype(pa.int64()),
    pa.uint8(): pd.ArrowDtype(pa.uint8()),
    pa.uint16(): pd.ArrowDtype(pa.uint16()),
    pa.uint32(): pd.ArrowDtype(pa.uint32()),
    pa.uint64(): pd.ArrowDtype(pa.uint64()),
    pa.float16(): pd.ArrowDtype(pa.float16()),
    pa.float32(): pd.ArrowDtype(pa.float32()),
    pa.float64(): pd.ArrowDtype(pa.float64()),
    pa.time32("s"): pd.ArrowDtype(pa.time32("s")),
    pa.time64("us"): pd.ArrowDtype(pa.time64("us")),
    pa.time64("ns"): pd.ArrowDtype(pa.time64("ns")),
    pa.timestamp("s"): pd.ArrowDtype(pa.timestamp("s")),
    pa.timestamp("ms"): pd.ArrowDtype(pa.timestamp("ms")),
    pa.timestamp("us"): pd.ArrowDtype(pa.timestamp("us")),
    pa.timestamp("ns"): pd.ArrowDtype(pa.timestamp("ns")),
    pa.timestamp("s", "UTC"): pd.ArrowDtype(pa.timestamp("s", "UTC")),
    pa.timestamp("ms", "UTC"): pd.ArrowDtype(pa.timestamp("ms", "UTC")),
    pa.timestamp("us", "UTC"): pd.ArrowDtype(pa.timestamp("us", "UTC")),
    pa.timestamp("ns", "UTC"): pd.ArrowDtype(pa.timestamp("ns", "UTC")),
    pa.date32(): pd.ArrowDtype(pa.date32()),
    pa.date64(): pd.ArrowDtype(pa.date64()),
    pa.binary(): pd.ArrowDtype(pa.binary()),
    pa.string(): pd.ArrowDtype(pa.string()),
    pa.large_string(): pd.ArrowDtype(pa.string()),  # map it back to panda's string[pyarrow]
}


def is_string_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_string(x.dtype.pyarrow_dtype) or pa.types.is_large_string(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_string_dtype(x)


def is_integer_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_integer(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_integer_dtype(x)


def is_float_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_floating(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_float_dtype(x)


def is_date_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_date(x.dtype.pyarrow_dtype)
    else:
        return False


def is_timestamp_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_timestamp(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_datetime64_any_dtype(x)


def is_boolean_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_boolean(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_bool_dtype(x)
