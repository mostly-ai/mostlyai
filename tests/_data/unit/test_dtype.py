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

import datetime

import pandas as pd
import pyarrow as pa
import pytest

from mostlyai.sdk._data.dtype import (
    BOOL,
    FLOAT64,
    INT64,
    STRING,
    VirtualBoolean,
    VirtualFloat,
    VirtualInteger,
    VirtualVarchar,
    bool_coerce,
    datetime_coerce,
    float_coerce,
    int_coerce,
    str_coerce,
    time_coerce,
)


class TestEncompass:
    @pytest.fixture
    def bool_(self):
        yield pd.Series([True, False, False, True, True, None, True], dtype=BOOL)

    @pytest.fixture
    def varchar_8(self):
        yield pd.Series(
            ["abc", "abcdefgh", None, "cdef", "fghijkl", 123, "h"],
            dtype=STRING,
        )

    @pytest.fixture
    def int_len_4(self):
        yield pd.Series([-321, 12, pd.NA, 0.0], dtype=INT64)

    @pytest.fixture
    def float_len_8(self):
        yield pd.Series([1, 2, 123456, pd.NA, -12345, 123456.7], dtype=FLOAT64)

    def test_encompasses_varchar(self, varchar_8, int_len_4, float_len_8):
        assert not VirtualVarchar(length=7).encompasses(varchar_8)
        assert VirtualVarchar(length=8).encompasses(varchar_8)
        assert not VirtualVarchar(length=3).encompasses(int_len_4)
        assert VirtualVarchar(length=4).encompasses(int_len_4)
        assert not VirtualVarchar(length=7).encompasses(float_len_8)
        assert VirtualVarchar(length=8).encompasses(float_len_8)
        assert VirtualVarchar(length=10).encompasses(float_len_8)
        assert VirtualVarchar().encompasses(float_len_8)

    def test_encompass_varchar(self, varchar_8, int_len_4, float_len_8):
        assert VirtualVarchar(length=0).encompass(varchar_8) == VirtualVarchar(length=8)
        assert VirtualVarchar().encompass(varchar_8) == VirtualVarchar()
        assert VirtualVarchar(length=0).encompass(int_len_4) == VirtualVarchar(length=4)
        assert VirtualVarchar(length=0).encompass(float_len_8) == VirtualVarchar(length=8)

    def test_encompasses_boolean(self, int_len_4, float_len_8, bool_):
        assert VirtualBoolean().encompasses(bool_)
        assert not VirtualBoolean().encompasses(float_len_8)
        assert not VirtualBoolean().encompasses(int_len_4)

    def test_encompasses_integer(self, int_len_4, float_len_8, varchar_8):
        assert VirtualInteger().encompasses(int_len_4)
        assert not VirtualInteger().encompasses(float_len_8)
        assert not VirtualInteger().encompasses(varchar_8)

    def test_encompass_integer(self, int_len_4, float_len_8, varchar_8):
        assert VirtualInteger().encompass(int_len_4) == VirtualInteger()
        assert VirtualInteger().encompass(float_len_8) == VirtualFloat()
        assert VirtualInteger().encompass(varchar_8) == VirtualVarchar()

    def test_encompasses_float(self, int_len_4, float_len_8, varchar_8):
        assert VirtualFloat().encompasses(int_len_4)
        assert VirtualFloat().encompasses(float_len_8)
        assert not VirtualFloat().encompasses(varchar_8)

    def test_encompass_float(self, int_len_4, float_len_8, varchar_8):
        assert VirtualFloat().encompass(int_len_4) == VirtualFloat()
        assert VirtualFloat().encompass(float_len_8) == VirtualFloat()
        assert VirtualFloat().encompass(varchar_8) == VirtualVarchar()


class TestCoerce:
    def test_bool_coerce(self):
        def _assert(s1, s2):
            pd.testing.assert_series_equal(s1, s2)

        def _s(*elems, dtype=None):
            return pd.Series(elems, dtype=dtype)

        def _bool_s(*elems):
            return _s(*elems, dtype=BOOL)

        # nones
        _assert(bool_coerce(_s(None)), _bool_s(pd.NA))

        # strings
        _assert(bool_coerce(_s("abcde")), _bool_s(pd.NA))
        _assert(
            bool_coerce(_s("True", "False", dtype=pd.ArrowDtype(pa.string()))),
            _bool_s(True, False),
        )
        _assert(bool_coerce(_s("abcde", None)), _bool_s(pd.NA, pd.NA))

        # integers
        _assert(bool_coerce(_s(1)), _bool_s(True))
        _assert(bool_coerce(_s(0)), _bool_s(False))
        _assert(bool_coerce(_s(1, None, dtype=INT64)), _bool_s(True, pd.NA))
        _assert(bool_coerce(_s(2, None, dtype=INT64)), _bool_s(pd.NA, pd.NA))

        # floats
        _assert(bool_coerce(_s(1.0)), _bool_s(True))
        _assert(bool_coerce(_s(0.0)), _bool_s(False))
        _assert(bool_coerce(_s(1.0, None)), _bool_s(True, pd.NA))
        _assert(bool_coerce(_s(2.0, None)), _bool_s(pd.NA, pd.NA))

        # datetimes
        _assert(bool_coerce(_s(pd.to_datetime("01-02-2003"))), _bool_s(pd.NA))
        _assert(
            bool_coerce(_s(pd.to_datetime("01-02-2003"), pd.NA)),
            _bool_s(pd.NA, pd.NA),
        )

        # times
        _assert(bool_coerce(_s(datetime.time(10, 20))), _bool_s(pd.NA))
        _assert(bool_coerce(_s(datetime.time(10, 20), None)), _bool_s(pd.NA, pd.NA))

        # booleans
        _assert(bool_coerce(_s(True)), _bool_s(True))
        _assert(bool_coerce(_s(False)), _bool_s(False))
        _assert(bool_coerce(_s(True, None)), _bool_s(True, pd.NA))

    def test_datetime_coerce(self):
        def _assert(s1, s2):
            pd.testing.assert_series_equal(s1, s2)

        def _s(*elems, dtype=None):
            return pd.Series(elems, dtype=dtype)

        def _dt_s(*elems):
            return _s(*elems, dtype="datetime64[ns]")

        # TODO: timezones support

        # nones
        _assert(datetime_coerce(_s(None)), _dt_s(pd.NA))

        # strings
        _assert(datetime_coerce(_s("01-02-2003")), _dt_s("01-02-2003"))
        _assert(datetime_coerce(_s("abcde")), _dt_s(pd.NA))
        _assert(datetime_coerce(_s("01-02-2003", None)), _dt_s("01-02-2003", pd.NA))

        # integers
        _assert(datetime_coerce(_s(0)), _dt_s("1970-01-01"))
        _assert(datetime_coerce(_s(0, None, dtype=INT64)), _dt_s("1970-01-01", pd.NA))

        # floats
        _assert(datetime_coerce(_s(0.999)), _dt_s("1970-01-01"))
        _assert(datetime_coerce(_s(0.999, None)), _dt_s("1970-01-01", pd.NA))

        # datetimes
        _assert(datetime_coerce(_s(pd.to_datetime("01-02-2003"))), _dt_s("01-02-2003"))
        _assert(
            datetime_coerce(_s(pd.to_datetime("01-02-2003"), None)),
            _dt_s("01-02-2003", pd.NA),
        )

        # times
        _assert(datetime_coerce(_s(datetime.time(10, 20))), _dt_s(pd.NA))
        _assert(
            datetime_coerce(_s(datetime.time(10, 20), None)),
            _dt_s(pd.NA, pd.NA),
        )

        # booleans
        _assert(datetime_coerce(_s(True)), _dt_s(pd.NA))
        _assert(datetime_coerce(_s(False)), _dt_s(pd.NA))
        _assert(datetime_coerce(_s(True, None)), _dt_s(pd.NA, pd.NA))

    def test_float_coerce(self):
        def _assert(s1, s2):
            pd.testing.assert_series_equal(s1, s2)

        def _s(*elems, dtype=None):
            return pd.Series(elems, dtype=dtype)

        def _flt_s(*elems):
            return _s(*elems, dtype=FLOAT64)

        # nones
        _assert(float_coerce(_s(None)), _flt_s(pd.NA))

        # strings
        _assert(float_coerce(_s("1")), _flt_s(1.0))
        _assert(float_coerce(_s("1.1")), _flt_s(1.1))
        _assert(float_coerce(_s("01-02-2003")), _flt_s(pd.NA))
        _assert(float_coerce(_s("abcde")), _flt_s(pd.NA))
        _assert(float_coerce(_s("1.1", None)), _flt_s(1.1, pd.NA))

        # integers
        _assert(float_coerce(_s(0)), _flt_s(0.0))
        _assert(float_coerce(_s(0, None, dtype=INT64)), _flt_s(0.0, pd.NA))

        # floats
        _assert(float_coerce(_s(0.999)), _flt_s(0.999))
        _assert(float_coerce(_s(0.999, None)), _flt_s(0.999, pd.NA))

        # datetimes
        _assert(float_coerce(_s(pd.to_datetime("1970-01-01"))), _flt_s(0.0))
        _assert(
            float_coerce(_s(pd.to_datetime("1970-01-01"), None)),
            _flt_s(0, -9.223372036854776e18),
        )  # TODO: understand the behaviour

        # times
        _assert(float_coerce(_s(datetime.time(10, 20))), _flt_s(pd.NA))
        _assert(float_coerce(_s(datetime.time(10, 20), None)), _flt_s(pd.NA, pd.NA))

        # booleans
        _assert(float_coerce(_s(True)), _flt_s(1.0))
        _assert(float_coerce(_s(False)), _flt_s(0.0))
        _assert(float_coerce(_s(True, None)), _flt_s(1.0, pd.NA))

    def test_int_coerce(self):
        def _assert(s1, s2):
            pd.testing.assert_series_equal(s1, s2)

        def _s(*elems, dtype=None):
            return pd.Series(elems, dtype=dtype)

        def _int_s(*elems):
            return _s(*elems, dtype=INT64)

        # nones
        _assert(int_coerce(_s(None)), _int_s(pd.NA))

        # strings
        _assert(int_coerce(_s("1")), _int_s(1))
        _assert(int_coerce(_s("1.1")), _int_s(1))
        _assert(int_coerce(_s("01-02-2003")), _int_s(pd.NA))
        _assert(int_coerce(_s("abcde")), _int_s(pd.NA))
        _assert(int_coerce(_s("1", None)), _int_s(1, pd.NA))

        # integers
        _assert(int_coerce(_s(0)), _int_s(0))
        _assert(int_coerce(_s(0, None, dtype=INT64)), _int_s(0, pd.NA))

        # floats
        _assert(int_coerce(_s(0.999)), _int_s(0))
        _assert(int_coerce(_s(0.999, None)), _int_s(0, pd.NA))

        # datetimes
        _assert(int_coerce(_s(pd.to_datetime("1970-01-01"))), _int_s(0))
        _assert(
            int_coerce(_s(pd.to_datetime("1970-01-01"), None)),
            _int_s(0, -9.223372036854776e18),
        )  # TODO: understand the behaviour

        # times
        _assert(int_coerce(_s(datetime.time(10, 20))), _int_s(pd.NA))
        _assert(int_coerce(_s(datetime.time(10, 20), None)), _int_s(pd.NA, pd.NA))

        # booleans
        _assert(int_coerce(_s(True)), _int_s(1))
        _assert(int_coerce(_s(False)), _int_s(0))
        _assert(int_coerce(_s(True, None)), _int_s(1, pd.NA))

    def test_str_coerce(self):
        def _assert(s1, s2):
            pd.testing.assert_series_equal(s1, s2)

        def _s(*elems, dtype=None):
            return pd.Series(elems, dtype=dtype)

        def _str_s(*elems):
            return _s(*elems, dtype=STRING)

        # nones
        _assert(str_coerce(_s(None), None), _str_s(pd.NA))

        # strings
        _assert(str_coerce(_s("abcde"), 5), _str_s("abcde"))
        _assert(str_coerce(_s("abcde"), 2), _str_s("ab"))
        _assert(str_coerce(_s("abcde", None), 2), _str_s("ab", pd.NA))

        # integers
        _assert(str_coerce(_s(12345), 5), _str_s("12345"))
        _assert(str_coerce(_s(12345), 3), _str_s("123"))
        _assert(
            str_coerce(_s(12345, None, dtype=INT64), 3),
            _str_s("123", pd.NA),
        )

        # floats
        _assert(str_coerce(_s(12345.1), 7), _str_s("12345.1"))
        _assert(str_coerce(_s(12345.1), 6), _str_s("12345."))
        _assert(str_coerce(_s(12345.1, None), 6), _str_s("12345.", pd.NA))

        # datetimes
        _assert(
            str_coerce(_s(pd.to_datetime("01-02-2003")), 9),
            _str_s("2003-01-0"),
        )
        _assert(
            str_coerce(_s(pd.to_datetime("01-02-2003")), 10),
            _str_s("2003-01-02"),
        )
        _assert(
            str_coerce(_s(pd.to_datetime("01-02-2003"), None), 10),
            _str_s("2003-01-02", pd.NA),
        )

        # times
        _assert(
            str_coerce(_s(datetime.time(10, 20)), 8),
            _str_s("10:20:00"),
        )
        _assert(str_coerce(_s(datetime.time(10, 20)), 7), _str_s("10:20:0"))
        _assert(
            str_coerce(_s(datetime.time(10, 20), None), 8),
            _str_s("10:20:00", pd.NA),
        )

        # booleans
        _assert(str_coerce(_s(True), 4), _str_s("True"))
        _assert(str_coerce(_s(True), 3), _str_s("Tru"))
        _assert(str_coerce(_s(True, None), 3), _str_s("Tru", pd.NA))

    def test_time_coerce(self):
        def _assert(s1, s2):
            pd.testing.assert_series_equal(s1, s2)

        def _s(*elems, dtype=None):
            return pd.Series(elems, dtype=dtype)

        def _t_s(*elems):
            return _s(*elems, dtype="object")

        _time = datetime.time.fromisoformat

        # nones
        _assert(time_coerce(_s(None)), _t_s(pd.NA))

        # strings
        _assert(time_coerce(_s("01-02-2003")), _t_s(_time("00:00:00")))
        _assert(time_coerce(_s("12:13")), _t_s(_time("12:13:00")))
        _assert(time_coerce(_s("abcde", None)), _t_s(pd.NA, pd.NA))

        # integers
        _assert(time_coerce(_s(0)), _t_s(pd.NA))
        _assert(time_coerce(_s(0, None, dtype=INT64)), _t_s(pd.NA, pd.NA))

        # floats
        _assert(time_coerce(_s(0.999)), _t_s(pd.NA))
        _assert(time_coerce(_s(0.999, None)), _t_s(pd.NA, pd.NA))

        # datetimes
        _assert(time_coerce(_s(pd.to_datetime("01-02-2003"))), _t_s(_time("00:00:00")))
        _assert(
            time_coerce(_s(pd.to_datetime("01-02-2003"), None)),
            _t_s(_time("00:00:00"), pd.NA),
        )

        # times
        _assert(time_coerce(_s(_time("10:20"))), _t_s(_time("10:20")))
        _assert(
            time_coerce(_s(_time("10:20"), None)),
            _t_s(_time("10:20"), pd.NA),
        )

        # booleans
        _assert(time_coerce(_s(True)), _t_s(pd.NA))
        _assert(time_coerce(_s(False)), _t_s(pd.NA))
        _assert(time_coerce(_s(True, None)), _t_s(pd.NA, pd.NA))
