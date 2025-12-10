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

"""unit tests for constraint transformations."""

import numpy as np
import pandas as pd
import pytest

from mostlyai.sdk._data.constraint_transformations import (
    ConstraintTranslator,
    FixedCombinationHandler,
    InequalityHandler,
    OneHotEncodingHandler,
    RangeHandler,
)
from mostlyai.sdk.domain import (
    FixedCombination,
    Generator,
    Inequality,
    ModelConfiguration,
    OneHotEncoding,
    Range,
    SourceColumn,
    SourceTable,
)


class TestFixedCombinationHandler:
    def test_to_internal_merges_columns(self):
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        df = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"], "value": [1, 2]})

        result = handler.to_internal(df)

        assert "state|city" in result.columns
        # internal representation uses record separator (\x1E), not "|"
        # verify round-trip works instead of checking internal format
        restored = handler.to_original(result)
        assert list(restored["state"]) == ["CA", "NY"]
        assert list(restored["city"]) == ["LA", "NYC"]
        assert "state" in result.columns
        assert "city" in result.columns

    def test_to_original_keeps_original_columns(self):
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        # simulate generation output: create internal format first
        df_input = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"], "value": [1, 2]})
        internal = handler.to_internal(df_input)
        # simulate generation output: original columns + merged column
        df = pd.DataFrame({col: internal[col] for col in internal.columns})

        result = handler.to_original(df)

        assert "state" in result.columns
        assert "city" in result.columns
        assert "state|city" not in result.columns
        assert list(result["state"]) == ["CA", "NY"]
        assert list(result["city"]) == ["LA", "NYC"]

    def test_to_original_splits_columns_fallback(self):
        # test fallback: if only merged column exists, split it
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        # create internal format with record separator
        internal = handler.to_internal(pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"]}))
        df = pd.DataFrame({col: internal[col] for col in ["state|city", "state", "city"] if col in internal.columns})
        # remove original columns to test fallback
        df = df.drop(columns=["state", "city"], errors="ignore")

        result = handler.to_original(df)

        assert "state" in result.columns
        assert "city" in result.columns
        assert "state|city" not in result.columns
        assert list(result["state"]) == ["CA", "NY"]
        assert list(result["city"]) == ["LA", "NYC"]

    def test_round_trip(self):
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b", "c"]))
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"], "c": ["!", "@"], "other": [10, 20]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert set(df.columns) == set(restored.columns)
        for col in df.columns:
            assert list(df[col]) == list(restored[col])

    def test_encoding_types(self):
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b"]))
        assert handler.get_encoding_types() == {"a|b": "TABULAR_CATEGORICAL"}

    def test_separator_character_in_data_escaping(self):
        """test that separator characters in data are properly escaped."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        # include the record separator character (\x1E) in the data
        df = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA\x1eSF", "NYC"], "value": [1, 2]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["state"]) == ["CA", "NY"]
        assert list(restored["city"]) == ["LA\x1eSF", "NYC"]

    def test_pipe_character_in_data(self):
        """test that pipe characters in data are preserved (not used as separator)."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        # pipe character in data should be preserved since we use record separator
        df = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA|SF", "NYC"], "value": [1, 2]})

        internal = handler.to_internal(df)
        # new format uses record separator, so pipe in data is preserved
        restored = handler.to_original(internal)

        assert list(restored["state"]) == ["CA", "NY"]
        assert list(restored["city"]) == ["LA|SF", "NYC"]

    def test_multiple_columns_with_separator(self):
        """test escaping with multiple columns containing separator."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b", "c"]))
        df = pd.DataFrame({"a": ["x\x1ey", "z"], "b": ["1", "2\x1e3"], "c": ["!", "@"], "other": [10, 20]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["a"]) == ["x\x1ey", "z"]
        assert list(restored["b"]) == ["1", "2\x1e3"]
        assert list(restored["c"]) == ["!", "@"]

    def test_malformed_split_handling(self):
        """test that malformed splits are handled gracefully."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b", "c"]))
        # create data that would split into wrong number of columns
        # this shouldn't happen in practice, but we should handle it
        df = pd.DataFrame({"a|b|c": ["x", "y|z"], "value": [1, 2]})

        # should not crash, but may produce incomplete data
        restored = handler.to_original(df)
        assert "a" in restored.columns
        assert "b" in restored.columns
        assert "c" in restored.columns


class TestInequalityHandler:
    def test_to_internal_numeric(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30], "end": [15, 25, 35]})

        result = handler.to_internal(df)

        assert "start" in result.columns
        delta_col = [c for c in result.columns if c.startswith("__constraint_ineq_delta")][0]
        assert list(result[delta_col]) == [5, 5, 5]

    def test_to_internal_datetime(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "end": pd.to_datetime(["2024-01-10", "2024-02-15"]),
            }
        )

        result = handler.to_internal(df)

        delta_col = [c for c in result.columns if c.startswith("__constraint_ineq_delta")][0]
        assert result[delta_col].iloc[0] == pd.Timedelta(days=9)
        assert result[delta_col].iloc[1] == pd.Timedelta(days=14)

    def test_to_original_numeric(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        delta_col = handler._delta_column
        df = pd.DataFrame({"start": [10, 20], delta_col: [5, 10]})

        result = handler.to_original(df)

        assert "end" in result.columns
        assert list(result["end"]) == [15, 30]
        assert delta_col not in result.columns

    def test_to_original_datetime(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        delta_col = handler._delta_column
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                delta_col: pd.to_timedelta(["9 days", "14 days"]),
            }
        )

        result = handler.to_original(df)

        assert result["end"].iloc[0] == pd.Timestamp("2024-01-10")
        assert result["end"].iloc[1] == pd.Timestamp("2024-02-15")

    def test_round_trip_numeric(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="low", high_column="high"))
        df = pd.DataFrame({"low": [1.0, 2.0, 3.0], "high": [5.0, 7.0, 10.0], "other": ["a", "b", "c"]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["low"]) == [1.0, 2.0, 3.0]
        assert list(restored["high"]) == [5.0, 7.0, 10.0]

    def test_round_trip_datetime(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-06-15"]),
                "end": pd.to_datetime(["2024-01-31", "2024-12-31"]),
            }
        )

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        pd.testing.assert_series_equal(restored["start"], df["start"])
        pd.testing.assert_series_equal(restored["end"], df["end"])

    def test_violation_correction(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="low", high_column="high"))
        df = pd.DataFrame({"low": [10, 20], "high": [5, 25]})  # first row violates

        internal = handler.to_internal(df)
        delta_col = handler._delta_column

        assert internal[delta_col].iloc[0] == 5  # corrected to abs
        assert internal[delta_col].iloc[1] == 5

    def test_encoding_types(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="a", high_column="b"))
        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_NUMERIC_AUTO"

    def test_strict_boundaries_false_allows_equality(self):
        """test that strict_boundaries=False (default) allows low == high."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end", strict_boundaries=False)
        )
        df = pd.DataFrame({"start": [10, 20], "end": [10, 25]})
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] == 0  # equality allowed
        assert result[handler._delta_column].iloc[1] == 5

    def test_strict_boundaries_enforces_strict_inequality(self):
        """test that strict_boundaries=True enforces low < high for numeric and datetime."""
        # numeric
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end", strict_boundaries=True)
        )
        df = pd.DataFrame({"start": [10, 20, 30], "end": [10, 25, 35]})
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] > 0
        assert list(result[handler._delta_column].iloc[1:]) == [5, 5]

        # datetime
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "end": pd.to_datetime(["2024-01-01", "2024-02-15"]),
            }
        )
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] > pd.Timedelta(0)
        assert result[handler._delta_column].iloc[1] == pd.Timedelta(days=14)

    def test_strict_boundaries_to_original(self):
        """test that to_original enforces strict inequality when strict_boundaries=True."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end", strict_boundaries=True)
        )
        delta_col = handler._delta_column

        # numeric
        df = pd.DataFrame({"start": [10, 20], delta_col: [0, 5]})
        result = handler.to_original(df)
        assert result["end"].iloc[0] > result["start"].iloc[0]
        assert result["end"].iloc[1] == 25

        # datetime
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                delta_col: pd.to_timedelta(["0 days", "14 days"]),
            }
        )
        result = handler.to_original(df)
        assert result["end"].iloc[0] > result["start"].iloc[0]
        assert result["end"].iloc[1] == pd.Timestamp("2024-02-15")

    def test_strict_boundaries_round_trip(self):
        """test round-trip with strict_boundaries=True preserves strict inequality and corrects equality."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="low", high_column="high", strict_boundaries=True)
        )

        # normal case
        df = pd.DataFrame({"low": [1.0, 2.0, 3.0], "high": [5.0, 7.0, 10.0]})
        internal = handler.to_internal(df)
        restored = handler.to_original(internal)
        assert all(restored["high"] > restored["low"])
        assert list(restored["low"]) == [1.0, 2.0, 3.0]
        assert list(restored["high"]) == [5.0, 7.0, 10.0]

        # with equality in input
        df = pd.DataFrame({"low": [1.0, 2.0], "high": [1.0, 3.0]})
        internal = handler.to_internal(df)
        restored = handler.to_original(internal)
        assert all(restored["high"] > restored["low"])

    def test_strict_boundaries_preserves_dtype(self):
        """test that strict_boundaries=True preserves integer dtype and uses appropriate epsilon."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end", strict_boundaries=True)
        )

        # integer input: uses 1, preserves integer dtype
        df = pd.DataFrame({"start": [10, 20, 30], "end": [10, 25, 35]}, dtype=int)
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] == 1
        assert pd.api.types.is_integer_dtype(result[handler._delta_column])
        assert list(result[handler._delta_column].iloc[1:]) == [5, 5]

        # integer round-trip
        internal = handler.to_internal(df)
        restored = handler.to_original(internal)
        assert pd.api.types.is_integer_dtype(restored["start"])
        assert pd.api.types.is_integer_dtype(restored["end"])
        assert list(restored["end"]) == [11, 25, 35]  # 10 + 1 = 11

        # integer delta in to_original
        delta_col = handler._delta_column
        df = pd.DataFrame({"start": [10, 20], delta_col: [0, 5]}, dtype=int)
        result = handler.to_original(df)
        assert result["end"].iloc[0] == 11
        assert pd.api.types.is_integer_dtype(result["end"])

        # float input: uses epsilon, preserves float dtype
        df = pd.DataFrame({"start": [10.5, 20.0], "end": [10.5, 25.0]}, dtype=float)
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] > 0
        assert isinstance(result[handler._delta_column].iloc[0], (float, np.floating))
        assert result[handler._delta_column].iloc[1] == 5.0

    def test_missing_columns_raises_error(self):
        """test that missing columns raise a clear error message."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20]})  # missing "end" column

        with pytest.raises(ValueError, match="columns.*not found in dataframe"):
            handler.to_internal(df)


class TestRangeHandler:
    def test_to_internal_numeric(self):
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [0, 10], "mid": [5, 15], "max": [10, 20]})

        result = handler.to_internal(df)

        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        assert list(result[delta1_col]) == [5, 5]
        assert list(result[delta2_col]) == [5, 5]

    def test_to_internal_datetime(self):
        handler = RangeHandler(
            Range(table_name="test_table", low_column="start", middle_column="middle", high_column="end")
        )
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01"]),
                "middle": pd.to_datetime(["2024-01-10"]),
                "end": pd.to_datetime(["2024-01-20"]),
            }
        )

        result = handler.to_internal(df)

        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        assert result[delta1_col].iloc[0] == pd.Timedelta(days=9)
        assert result[delta2_col].iloc[0] == pd.Timedelta(days=10)

    def test_to_original_numeric(self):
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        df = pd.DataFrame({"min": [0, 100], delta1_col: [5, 10], delta2_col: [5, 20]})

        result = handler.to_original(df)

        assert list(result["mid"]) == [5, 110]
        assert list(result["max"]) == [10, 130]

    def test_round_trip_numeric(self):
        handler = RangeHandler(Range(table_name="test_table", low_column="a", middle_column="b", high_column="c"))
        df = pd.DataFrame({"a": [0.0, 100.0], "b": [50.0, 150.0], "c": [100.0, 200.0]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["a"]) == [0.0, 100.0]
        assert list(restored["b"]) == [50.0, 150.0]
        assert list(restored["c"]) == [100.0, 200.0]

    def test_round_trip_datetime(self):
        handler = RangeHandler(
            Range(table_name="test_table", low_column="start", middle_column="middle", high_column="end")
        )
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-06-01"]),
                "middle": pd.to_datetime(["2024-03-01", "2024-09-01"]),
                "end": pd.to_datetime(["2024-06-01", "2024-12-01"]),
            }
        )

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        pd.testing.assert_series_equal(restored["start"], df["start"])
        pd.testing.assert_series_equal(restored["middle"], df["middle"])
        pd.testing.assert_series_equal(restored["end"], df["end"])

    def test_violation_correction(self):
        handler = RangeHandler(Range(table_name="test_table", low_column="a", middle_column="b", high_column="c"))
        df = pd.DataFrame({"a": [10], "b": [5], "c": [20]})  # b < a violates

        internal = handler.to_internal(df)

        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        assert internal[delta1_col].iloc[0] == 5  # corrected to abs
        assert internal[delta2_col].iloc[0] == 15  # corrected to abs

    def test_encoding_types(self):
        handler = RangeHandler(Range(table_name="test_table", low_column="a", middle_column="b", high_column="c"))
        enc = handler.get_encoding_types()
        assert len(enc) == 2
        assert all(v == "TABULAR_NUMERIC_AUTO" for v in enc.values())


class TestOneHotEncodingHandler:
    def test_to_internal_converts_to_categorical(self):
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["is_a", "is_b", "is_c"]))
        df = pd.DataFrame({"is_a": [1, 0, 0], "is_b": [0, 1, 0], "is_c": [0, 0, 1], "other": [10, 20, 30]})

        result = handler.to_internal(df)

        internal_col = handler._internal_column
        assert internal_col in result.columns
        assert list(result[internal_col]) == ["is_a", "is_b", "is_c"]
        # original columns are kept during to_internal
        assert "is_a" in result.columns

    def test_to_original_creates_onehot(self):
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["cat_a", "cat_b", "cat_c"]))
        internal_col = handler._internal_column
        df = pd.DataFrame({internal_col: ["cat_a", "cat_b", "cat_c"], "other": [1, 2, 3]})

        result = handler.to_original(df)

        assert "cat_a" in result.columns
        assert "cat_b" in result.columns
        assert "cat_c" in result.columns
        assert internal_col not in result.columns
        assert list(result["cat_a"]) == [1, 0, 0]
        assert list(result["cat_b"]) == [0, 1, 0]
        assert list(result["cat_c"]) == [0, 0, 1]

    def test_round_trip(self):
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["x", "y", "z"]))
        df = pd.DataFrame({"x": [1, 0, 0, 0], "y": [0, 1, 0, 0], "z": [0, 0, 1, 0], "value": [100, 200, 300, 400]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["x"]) == [1, 0, 0, 0]
        assert list(restored["y"]) == [0, 1, 0, 0]
        assert list(restored["z"]) == [0, 0, 1, 0]

    def test_encoding_types(self):
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["a", "b"]))
        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_CATEGORICAL"

    def test_handles_null_rows(self):
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["col_a", "col_b"]))
        internal_col = handler._internal_column
        df = pd.DataFrame({internal_col: ["col_a", None, "col_b"], "other": [1, 2, 3]})

        result = handler.to_original(df)

        assert list(result["col_a"]) == [1, 0, 0]
        assert list(result["col_b"]) == [0, 0, 1]

    def test_handles_all_zeros_row(self):
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["a", "b", "c"]))
        df = pd.DataFrame({"a": [1, 0], "b": [0, 0], "c": [0, 0], "other": [10, 20]})

        internal = handler.to_internal(df)

        internal_col = handler._internal_column
        assert internal[internal_col].iloc[0] == "a"
        assert internal[internal_col].iloc[1] is None


class TestConstraintTranslator:
    def test_mixed_constraints(self):
        constraints = [
            FixedCombination(table_name="test_table", columns=["state", "city"]),
            Inequality(table_name="test_table", low_column="start_age", high_column="end_age"),
        ]
        translator = ConstraintTranslator(constraints)
        df = pd.DataFrame(
            {
                "state": ["CA", "NY"],
                "city": ["LA", "NYC"],
                "start_age": [20, 30],
                "end_age": [25, 40],
                "other": [1, 2],
            }
        )

        internal = translator.to_internal(df)
        restored = translator.to_original(internal)

        assert set(df.columns) == set(restored.columns)
        for col in df.columns:
            assert list(df[col]) == list(restored[col])

    def test_get_internal_columns(self):
        constraints = [
            FixedCombination(table_name="test_table", columns=["a", "b"]),
            Inequality(table_name="test_table", low_column="low", high_column="high"),
        ]
        translator = ConstraintTranslator(constraints)
        original = ["a", "b", "low", "high", "other"]

        internal = translator.get_internal_columns(original)

        # FixedCombination keeps original columns alongside merged column
        assert "a" in internal
        assert "b" in internal
        assert "a|b" in internal
        # Inequality removes high column
        assert "high" not in internal
        assert "low" in internal
        assert "other" in internal
        assert any(c.startswith("__constraint_ineq_delta") for c in internal)

    def test_get_original_columns(self):
        constraints = [FixedCombination(table_name="test_table", columns=["a", "b"])]
        translator = ConstraintTranslator(constraints)
        internal = ["a|b", "other"]

        original = translator.get_original_columns(internal)

        assert original == ["a", "b", "other"]

    def test_get_encoding_types(self):
        constraints = [
            FixedCombination(table_name="test_table", columns=["a", "b"]),
            Range(table_name="test_table", low_column="x", middle_column="y", high_column="z"),
        ]
        translator = ConstraintTranslator(constraints)

        enc = translator.get_encoding_types()

        assert enc["a|b"] == "TABULAR_CATEGORICAL"
        assert sum(1 for v in enc.values() if v == "TABULAR_NUMERIC_AUTO") == 2

    def test_from_generator_config_with_constraints(self):
        generator = Generator(
            id="test-gen",
            name="Test Generator",
            tables=[
                SourceTable(
                    name="customers",
                    columns=[
                        SourceColumn(name="id"),
                        SourceColumn(name="state"),
                        SourceColumn(name="city"),
                    ],
                    tabular_model_configuration=ModelConfiguration(),
                )
            ],
            constraints=[FixedCombination(table_name="customers", columns=["state", "city"])],
        )

        translator, original_columns = ConstraintTranslator.from_generator_config(
            generator=generator, table_name="customers"
        )

        assert translator is not None
        assert len(translator.handlers) == 1
        assert original_columns == ["id", "state", "city"]

    def test_from_generator_config_no_constraints(self):
        generator = Generator(
            id="test-gen",
            name="Test Generator",
            tables=[SourceTable(name="table", columns=[SourceColumn(name="col1")])],
        )

        translator, columns = ConstraintTranslator.from_generator_config(generator=generator, table_name="table")

        assert translator is None
        assert columns is None

    def test_from_generator_config_table_not_found(self):
        generator = Generator(
            id="test-gen",
            name="Test Generator",
            tables=[SourceTable(name="existing", columns=[SourceColumn(name="col1")])],
        )

        translator, columns = ConstraintTranslator.from_generator_config(generator=generator, table_name="nonexistent")

        assert translator is None
        assert columns is None

    def test_from_generator_config_language_model(self):
        generator = Generator(
            id="test-gen",
            name="Test Generator",
            tables=[
                SourceTable(
                    name="docs",
                    columns=[SourceColumn(name="country"), SourceColumn(name="language")],
                    language_model_configuration=ModelConfiguration(),
                )
            ],
            constraints=[FixedCombination(table_name="docs", columns=["country", "language"])],
        )

        translator, original_columns = ConstraintTranslator.from_generator_config(
            generator=generator, table_name="docs"
        )

        assert translator is not None
        assert original_columns == ["country", "language"]


class TestDomainValidation:
    def test_fixed_combination_requires_two_columns(self):
        with pytest.raises(ValueError, match="at least 2 columns, got 1"):
            FixedCombination(table_name="test_table", columns=["single"])

    def test_inequality_same_column_fails(self):
        with pytest.raises(ValueError, match="must be different, both are 'col'"):
            Inequality(table_name="test_table", low_column="col", high_column="col")

    def test_range_duplicate_columns_fails(self):
        with pytest.raises(ValueError, match="must all be different"):
            Range(table_name="test_table", low_column="a", middle_column="a", high_column="b")

        with pytest.raises(ValueError, match="must all be different"):
            Range(table_name="test_table", low_column="a", middle_column="b", high_column="a")

    def test_onehot_requires_two_columns(self):
        with pytest.raises(ValueError, match="at least 2 columns, got 1"):
            OneHotEncoding(table_name="test_table", columns=["single"])

    def test_valid_constraints_create(self):
        fc = FixedCombination(table_name="test_table", columns=["a", "b", "c"])
        assert fc.columns == ["a", "b", "c"]

        ineq = Inequality(table_name="test_table", low_column="start", high_column="end")
        assert ineq.low_column == "start"

        rng = Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max")
        assert rng.middle_column == "mid"

        ohe = OneHotEncoding(table_name="test_table", columns=["is_x", "is_y"])
        assert ohe.columns == ["is_x", "is_y"]


class TestSeedDataPreservation:
    """test that seed data values are preserved during to_original transformation."""

    def test_fixed_combination_preserves_seed_data(self):
        """test that FixedCombination preserves seed values."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        df = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"], "state|city": ["merged1", "merged2"]})
        seed_data = pd.DataFrame({"state": ["TX", "FL"], "city": ["Houston", "Miami"]})

        result = handler.to_original(df, seed_data=seed_data)

        # seed values should be preserved
        assert list(result["state"]) == ["TX", "FL"]
        assert list(result["city"]) == ["Houston", "Miami"]
        assert "state|city" not in result.columns

    def test_inequality_preserves_seed_data(self):
        """test that Inequality preserves seed values."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20], handler._delta_column: [5, 10]})
        seed_data = pd.DataFrame({"start": [100, 200], "end": [150, 250]})

        result = handler.to_original(df, seed_data=seed_data)

        # both seed values should be preserved
        assert list(result["start"]) == [100, 200]
        assert list(result["end"]) == [150, 250]
        assert handler._delta_column not in result.columns

    def test_inequality_preserves_partial_seed_data(self):
        """test that Inequality preserves high_column seed value and reconstructs low_column from delta."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20], handler._delta_column: [5, 10]})
        seed_data = pd.DataFrame({"end": [150, 250]})  # only high_column seeded

        result = handler.to_original(df, seed_data=seed_data)

        # high_column seed value should be preserved, start should be reconstructed from delta: start = end - delta
        assert list(result["start"]) == [145, 240]  # 150 - 5, 250 - 10
        assert list(result["end"]) == [150, 250]  # from seed
        assert handler._delta_column not in result.columns

    def test_inequality_partial_seed_high_column(self):
        """test that Inequality reconstructs low_column from delta when high_column is partially seeded."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        # df has 4 rows, seed has 2 rows
        df = pd.DataFrame({"start": [10, 20, 30, 40], handler._delta_column: [5, 10, 15, 20]})
        seed_data = pd.DataFrame({"end": [150, 250]})  # only first 2 rows seeded

        result = handler.to_original(df, seed_data=seed_data)

        # first 2 rows: end from seed, start reconstructed from delta
        assert result["end"].iloc[0] == 150  # from seed
        assert result["end"].iloc[1] == 250  # from seed
        assert result["start"].iloc[0] == 145  # 150 - 5
        assert result["start"].iloc[1] == 240  # 250 - 10
        # last 2 rows: normal reconstruction (end = start + delta)
        assert result["end"].iloc[2] == 45  # 30 + 15
        assert result["end"].iloc[3] == 60  # 40 + 20
        assert result["start"].iloc[2] == 30  # from df
        assert result["start"].iloc[3] == 40  # from df
        assert handler._delta_column not in result.columns

    def test_inequality_imputation_partial_nulls(self):
        """test that Inequality handles row-by-row imputation with partial nulls per column."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        # df contains all columns: start, delta, and end (original columns are kept)
        # pattern: start = 10,20,30,40,50,60,70,80,90; delta = 5 for all; end = start + delta
        df = pd.DataFrame(
            {
                "start": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                handler._delta_column: [5, 5, 5, 5, 5, 5, 5, 5, 5],
                "end": [15, 25, 35, 45, 55, 65, 75, 85, 95],  # original values (will be overwritten by reconstruction)
            }
        )
        # seed data with partial nulls: covers all scenarios including constraint violations
        # rows 0-6: normal cases (both, only start, only end, neither)
        # rows 7-8: constraint violations to test delta-based reconstruction
        seed_data = pd.DataFrame(
            {
                "start": [10, 20, None, None, 50, 60, None, 100, None],  # row 7: start=100 violates (100 > 85)
                "end": [15, None, 35, 45, None, None, None, 85, 5],  # row 8: end=5 violates (5 < 90)
            }
        )

        result = handler.to_original(df, seed_data=seed_data)

        # expected results: [start, end] for each row
        expected = [
            [10, 15],  # both seeded (valid)
            [20, 25],  # only start: end = 20 + 5
            [30, 35],  # only end: start = 35 - 5
            [40, 45],  # only end: start = 45 - 5
            [50, 55],  # only start: end = 50 + 5
            [60, 65],  # only start: end = 60 + 5
            [70, 75],  # neither: end = 70 + 5
            [100, 85],  # both seeded but violates constraint (100 > 85) - preserved as-is, delta ignored
            [0, 5],  # only end seeded but violates (5 < 90) - start reconstructed as 5 - 5 = 0 using delta
        ]
        assert list(result["start"]) == [r[0] for r in expected]
        assert list(result["end"]) == [r[1] for r in expected]
        assert handler._delta_column not in result.columns

    def test_inequality_imputation_only_low_seeded(self):
        """test that Inequality handles imputation when only low_column has partial nulls."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30], handler._delta_column: [5, 10, 15]})
        # only start has some seeded values
        seed_data = pd.DataFrame({"start": [100, None, 300]})  # rows 0 and 2 seeded

        result = handler.to_original(df, seed_data=seed_data)

        # row 0: start seeded -> use seeded value, reconstruct end = 100 + 5 = 105
        assert result["start"].iloc[0] == 100
        assert result["end"].iloc[0] == 105

        # row 1: start not seeded -> use generated value, reconstruct end = 20 + 10 = 30
        assert result["start"].iloc[1] == 20
        assert result["end"].iloc[1] == 30

        # row 2: start seeded -> use seeded value, reconstruct end = 300 + 15 = 315
        assert result["start"].iloc[2] == 300
        assert result["end"].iloc[2] == 315

    def test_inequality_imputation_only_high_seeded(self):
        """test that Inequality handles imputation when only high_column has partial nulls."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30], handler._delta_column: [5, 10, 15]})
        # only end has some seeded values
        seed_data = pd.DataFrame({"end": [150, None, 350]})  # rows 0 and 2 seeded

        result = handler.to_original(df, seed_data=seed_data)

        # row 0: end seeded -> use seeded value, reconstruct start = 150 - 5 = 145
        assert result["start"].iloc[0] == 145
        assert result["end"].iloc[0] == 150

        # row 1: end not seeded -> reconstruct end = 20 + 10 = 30
        assert result["start"].iloc[1] == 20
        assert result["end"].iloc[1] == 30

        # row 2: end seeded -> use seeded value, reconstruct start = 350 - 15 = 335
        assert result["start"].iloc[2] == 335
        assert result["end"].iloc[2] == 350

    def test_range_preserves_seed_data(self):
        """test that Range preserves seed values."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [10, 20], handler._delta1_column: [5, 10], handler._delta2_column: [3, 5]})
        seed_data = pd.DataFrame({"min": [100, 200], "mid": [150, 250], "max": [180, 280]})

        result = handler.to_original(df, seed_data=seed_data)

        # all seed values should be preserved
        assert list(result["min"]) == [100, 200]
        assert list(result["mid"]) == [150, 250]
        assert list(result["max"]) == [180, 280]
        assert handler._delta1_column not in result.columns
        assert handler._delta2_column not in result.columns

    def test_range_seed_data_fewer_rows(self):
        """test that Range handles seed_data with fewer rows than df (pads with NaN)."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        # df has 3 rows
        df = pd.DataFrame({"min": [10, 20, 30], handler._delta1_column: [5, 10, 15], handler._delta2_column: [3, 5, 7]})
        # seed_data has only 1 row
        seed_data = pd.DataFrame({"min": [100], "mid": [150], "max": [180]})

        result = handler.to_original(df, seed_data=seed_data)

        # row 0: seeded values preserved
        assert result["min"].iloc[0] == 100
        assert result["mid"].iloc[0] == 150
        assert result["max"].iloc[0] == 180
        # row 1 and 2: reconstructed from delta (not seeded)
        assert result["min"].iloc[1] == 20  # original low value
        assert result["mid"].iloc[1] == 30  # 20 + 10
        assert result["max"].iloc[1] == 35  # 20 + 10 + 5
        assert result["min"].iloc[2] == 30  # original low value
        assert result["mid"].iloc[2] == 45  # 30 + 15
        assert result["max"].iloc[2] == 52  # 30 + 15 + 7
        assert handler._delta1_column not in result.columns
        assert handler._delta2_column not in result.columns

    def test_range_seed_data_more_rows(self):
        """test that Range handles seed_data with more rows than df (truncates)."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        # df has 2 rows
        df = pd.DataFrame({"min": [10, 20], handler._delta1_column: [5, 10], handler._delta2_column: [3, 5]})
        # seed_data has 5 rows
        seed_data = pd.DataFrame(
            {"min": [100, 200, 300, 400, 500], "mid": [150, 250, 350, 450, 550], "max": [180, 280, 380, 480, 580]}
        )

        result = handler.to_original(df, seed_data=seed_data)

        # only first 2 rows of seed_data should be used
        assert list(result["min"]) == [100, 200]
        assert list(result["mid"]) == [150, 250]
        assert list(result["max"]) == [180, 280]
        assert len(result) == 2
        assert handler._delta1_column not in result.columns
        assert handler._delta2_column not in result.columns

    def test_range_partial_seed_with_nan(self):
        """test that Range handles partial seeding with NaN values row-by-row."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [10, 20, 30], handler._delta1_column: [5, 10, 15], handler._delta2_column: [3, 5, 7]})
        # seed_data has NaN in some positions
        seed_data = pd.DataFrame(
            {
                "min": [100, np.nan, 300],  # row 1 not seeded
                "mid": [np.nan, 250, np.nan],  # only row 1 seeded
                "max": [180, np.nan, np.nan],  # only row 0 seeded
            }
        )

        result = handler.to_original(df, seed_data=seed_data)

        # row 0: min and max seeded, mid reconstructed
        assert result["min"].iloc[0] == 100
        assert result["mid"].iloc[0] == 105  # 100 + 5
        assert result["max"].iloc[0] == 180
        # row 1: mid seeded, min and max reconstructed
        assert result["min"].iloc[1] == 20  # original
        assert result["mid"].iloc[1] == 250  # seeded
        assert result["max"].iloc[1] == 35  # 20 + 10 + 5
        # row 2: only min seeded, mid and max reconstructed
        assert result["min"].iloc[2] == 300
        assert result["mid"].iloc[2] == 315  # 300 + 15
        assert result["max"].iloc[2] == 322  # 300 + 15 + 7
        assert handler._delta1_column not in result.columns
        assert handler._delta2_column not in result.columns

    def test_onehot_preserves_seed_data(self):
        """test that OneHotEncoding preserves seed values."""
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["is_a", "is_b", "is_c"]))
        df = pd.DataFrame({handler._internal_column: ["is_a", "is_b"]})
        seed_data = pd.DataFrame({"is_a": [1, 0], "is_b": [0, 1], "is_c": [0, 0]})

        result = handler.to_original(df, seed_data=seed_data)

        # seed values should be preserved
        assert list(result["is_a"]) == [1, 0]
        assert list(result["is_b"]) == [0, 1]
        assert list(result["is_c"]) == [0, 0]
        assert handler._internal_column not in result.columns

    def test_translator_preserves_seed_data(self):
        """test that ConstraintTranslator passes seed_data to handlers."""
        constraints = [
            FixedCombination(table_name="test_table", columns=["state", "city"]),
            Inequality(table_name="test_table", low_column="start", high_column="end"),
        ]
        translator = ConstraintTranslator(constraints)
        df = pd.DataFrame(
            {
                "state": ["CA", "NY"],
                "city": ["LA", "NYC"],
                "state|city": ["merged1", "merged2"],
                "start": [10, 20],
                translator.handlers[1]._delta_column: [5, 10],
            }
        )
        seed_data = pd.DataFrame(
            {"state": ["TX", "FL"], "city": ["Houston", "Miami"], "start": [100, 200], "end": [150, 250]}
        )

        result = translator.to_original(df, seed_data=seed_data)

        # seed values should be preserved for both constraints
        assert list(result["state"]) == ["TX", "FL"]
        assert list(result["city"]) == ["Houston", "Miami"]
        assert list(result["start"]) == [100, 200]
        assert list(result["end"]) == [150, 250]


class TestEdgeCases:
    """test edge cases and simple scenarios that may have been missed."""

    def test_inequality_empty_dataframe(self):
        """test that empty dataframes are handled gracefully."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": pd.Series([], dtype=float), "end": pd.Series([], dtype=float)})

        result = handler.to_internal(df)
        assert len(result) == 0
        assert handler._delta_column in result.columns

        restored = handler.to_original(result)
        assert len(restored) == 0
        assert "start" in restored.columns
        assert "end" in restored.columns

    def test_inequality_single_row(self):
        """test single row case."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10], "end": [20]})

        internal = handler.to_internal(df)
        assert len(internal) == 1

        restored = handler.to_original(internal)
        assert len(restored) == 1
        assert restored["start"].iloc[0] == 10
        assert restored["end"].iloc[0] == 20

    def test_fixed_combination_empty_strings(self):
        """test fixed combination with empty string values."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b"]))
        df = pd.DataFrame({"a": ["", "x", ""], "b": ["y", "", "z"]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["a"]) == ["", "x", ""]
        assert list(restored["b"]) == ["y", "", "z"]

    def test_fixed_combination_empty_dataframe(self):
        """test that empty dataframes work for FixedCombination."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b"]))
        df = pd.DataFrame({"a": pd.Series([], dtype=str), "b": pd.Series([], dtype=str)})

        internal = handler.to_internal(df)
        assert len(internal) == 0
        assert handler.merged_name in internal.columns

        restored = handler.to_original(internal)
        assert len(restored) == 0

    def test_fixed_combination_single_row(self):
        """test FixedCombination with single row."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b"]))
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["a"]) == ["x"]
        assert list(restored["b"]) == ["y"]

    def test_range_empty_dataframe(self):
        """test Range with empty dataframe."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame(
            {"min": pd.Series([], dtype=float), "mid": pd.Series([], dtype=float), "max": pd.Series([], dtype=float)}
        )

        internal = handler.to_internal(df)
        assert len(internal) == 0

        restored = handler.to_original(internal)
        assert len(restored) == 0

    def test_range_single_row(self):
        """test Range with single row."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [10], "mid": [20], "max": [30]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert restored["min"].iloc[0] == 10
        assert restored["mid"].iloc[0] == 20
        assert restored["max"].iloc[0] == 30

    def test_onehot_empty_dataframe(self):
        """test OneHotEncoding with empty dataframe."""
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["a", "b", "c"]))
        df = pd.DataFrame({"a": pd.Series([], dtype=int), "b": pd.Series([], dtype=int), "c": pd.Series([], dtype=int)})

        internal = handler.to_internal(df)
        assert len(internal) == 0

        restored = handler.to_original(internal)
        assert len(restored) == 0

    def test_onehot_single_row(self):
        """test OneHotEncoding with single row."""
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["a", "b"]))
        df = pd.DataFrame({"a": [1], "b": [0]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert restored["a"].iloc[0] == 1
        assert restored["b"].iloc[0] == 0

    def test_all_seeded_inequality(self):
        """test Inequality when all constraint columns are fully seeded."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30], handler._delta_column: [5, 10, 15]})
        # all rows seeded for both columns
        seed_data = pd.DataFrame({"start": [100, 200, 300], "end": [150, 250, 350]})

        result = handler.to_original(df, seed_data=seed_data)

        # all values should come from seed
        assert list(result["start"]) == [100, 200, 300]
        assert list(result["end"]) == [150, 250, 350]

    def test_all_seeded_fixed_combination(self):
        """test FixedCombination when all columns are fully seeded."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b"]))
        # create internal representation
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"], "a|b": ["merged1", "merged2"]})
        seed_data = pd.DataFrame({"a": ["seeded_a", "seeded_b"], "b": ["seeded_1", "seeded_2"]})

        result = handler.to_original(df, seed_data=seed_data)

        assert list(result["a"]) == ["seeded_a", "seeded_b"]
        assert list(result["b"]) == ["seeded_1", "seeded_2"]

    def test_all_seeded_range(self):
        """test Range when all columns are fully seeded."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [10, 20], handler._delta1_column: [5, 10], handler._delta2_column: [3, 5]})
        seed_data = pd.DataFrame({"min": [100, 200], "mid": [150, 250], "max": [180, 280]})

        result = handler.to_original(df, seed_data=seed_data)

        assert list(result["min"]) == [100, 200]
        assert list(result["mid"]) == [150, 250]
        assert list(result["max"]) == [180, 280]

    def test_inequality_with_none_seed_data(self):
        """test that None seed_data is handled correctly."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20], handler._delta_column: [5, 10]})

        result = handler.to_original(df, seed_data=None)

        assert list(result["start"]) == [10, 20]
        assert list(result["end"]) == [15, 30]

    def test_inequality_with_empty_seed_data(self):
        """test that empty seed_data is handled correctly."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20], handler._delta_column: [5, 10]})
        seed_data = pd.DataFrame()

        result = handler.to_original(df, seed_data=seed_data)

        assert list(result["start"]) == [10, 20]
        assert list(result["end"]) == [15, 30]

    def test_inequality_all_nan_values(self):
        """test Inequality with all NaN values in constraint columns."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [np.nan, np.nan], "end": [np.nan, np.nan]})

        # should not crash
        internal = handler.to_internal(df)
        assert len(internal) == 2
        assert handler._delta_column in internal.columns
        # delta should be NaN for NaN inputs
        assert internal[handler._delta_column].isna().all()

        restored = handler.to_original(internal)
        assert len(restored) == 2
        # NaN + NaN = NaN
        assert restored["end"].isna().all()

    def test_fixed_combination_all_nan_values(self):
        """test FixedCombination with all NaN values."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["a", "b"]))
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})

        internal = handler.to_internal(df)
        assert len(internal) == 2
        assert handler.merged_name in internal.columns

        restored = handler.to_original(internal)
        assert len(restored) == 2
        # NaN values are converted to empty strings in merge
        assert list(restored["a"]) == ["", ""]
        assert list(restored["b"]) == ["", ""]

    def test_range_all_nan_values(self):
        """test Range with all NaN values."""
        handler = RangeHandler(Range(table_name="test_table", low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [np.nan, np.nan], "mid": [np.nan, np.nan], "max": [np.nan, np.nan]})

        internal = handler.to_internal(df)
        assert len(internal) == 2

        restored = handler.to_original(internal)
        assert len(restored) == 2

    def test_onehot_multiple_ones_takes_first(self):
        """test OneHotEncoding with multiple 1s takes the first column."""
        handler = OneHotEncodingHandler(OneHotEncoding(table_name="test_table", columns=["a", "b", "c"]))
        # row 0: both a and b are 1 (invalid input)
        df = pd.DataFrame({"a": [1, 0], "b": [1, 1], "c": [0, 0]})

        internal = handler.to_internal(df)

        # should take first column with value 1
        assert internal[handler._internal_column].iloc[0] == "a"  # first 1 wins
        assert internal[handler._internal_column].iloc[1] == "b"

    def test_multiple_constraints_same_table(self):
        """test multiple constraints on the same table work together."""
        constraints = [
            FixedCombination(table_name="test_table", columns=["state", "city"]),
            Inequality(table_name="test_table", low_column="start_age", high_column="end_age"),
            Range(table_name="test_table", low_column="min_sal", middle_column="med_sal", high_column="max_sal"),
        ]
        translator = ConstraintTranslator(constraints)
        df = pd.DataFrame(
            {
                "state": ["CA", "NY"],
                "city": ["LA", "NYC"],
                "start_age": [20, 30],
                "end_age": [25, 40],
                "min_sal": [50000, 60000],
                "med_sal": [70000, 80000],
                "max_sal": [90000, 100000],
            }
        )

        internal = translator.to_internal(df)
        restored = translator.to_original(internal)

        # all values should be preserved
        assert list(restored["state"]) == ["CA", "NY"]
        assert list(restored["city"]) == ["LA", "NYC"]
        assert list(restored["start_age"]) == [20, 30]
        assert list(restored["end_age"]) == [25, 40]
        assert list(restored["min_sal"]) == [50000, 60000]
        assert list(restored["med_sal"]) == [70000, 80000]
        assert list(restored["max_sal"]) == [90000, 100000]

    def test_seed_data_length_mismatch_padding(self):
        """test that seed_data with fewer rows is properly padded."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30, 40], handler._delta_column: [5, 10, 15, 20]})
        # seed_data has fewer rows
        seed_data = pd.DataFrame({"start": [100], "end": [150]})

        result = handler.to_original(df, seed_data=seed_data)

        # first row from seed, rest reconstructed
        assert result["start"].iloc[0] == 100
        assert result["end"].iloc[0] == 150
        assert result["start"].iloc[1] == 20
        assert result["end"].iloc[1] == 30  # 20 + 10

    def test_seed_data_length_mismatch_truncation(self):
        """test that seed_data with more rows is properly truncated."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20], handler._delta_column: [5, 10]})
        # seed_data has more rows
        seed_data = pd.DataFrame({"start": [100, 200, 300, 400], "end": [150, 250, 350, 450]})

        result = handler.to_original(df, seed_data=seed_data)

        # only first 2 rows of seed used
        assert len(result) == 2
        assert list(result["start"]) == [100, 200]
        assert list(result["end"]) == [150, 250]

    def test_partial_nulls_in_seed_data(self):
        """test handling of partial nulls in seed_data columns."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30], handler._delta_column: [5, 10, 15]})
        # mix of seeded and null values
        seed_data = pd.DataFrame({"start": [100, np.nan, 300], "end": [np.nan, 250, 350]})

        result = handler.to_original(df, seed_data=seed_data)

        # row 0: start seeded, end reconstructed
        assert result["start"].iloc[0] == 100
        assert result["end"].iloc[0] == 105  # 100 + 5
        # row 1: end seeded, start reconstructed
        assert result["start"].iloc[1] == 240  # 250 - 10
        assert result["end"].iloc[1] == 250
        # row 2: both seeded
        assert result["start"].iloc[2] == 300
        assert result["end"].iloc[2] == 350
