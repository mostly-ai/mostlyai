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
        handler = FixedCombinationHandler(FixedCombination(columns=["state", "city"]))
        df = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"], "value": [1, 2]})

        result = handler.to_internal(df)

        assert "state|city" in result.columns
        assert list(result["state|city"]) == ["CA|LA", "NY|NYC"]
        assert "state" in result.columns
        assert "city" in result.columns

    def test_to_original_keeps_original_columns(self):
        handler = FixedCombinationHandler(FixedCombination(columns=["state", "city"]))
        # simulate generation output: original columns + merged column
        df = pd.DataFrame(
            {"state": ["CA", "NY"], "city": ["LA", "NYC"], "state|city": ["CA|LA", "NY|NYC"], "value": [1, 2]}
        )

        result = handler.to_original(df)

        assert "state" in result.columns
        assert "city" in result.columns
        assert "state|city" not in result.columns
        assert list(result["state"]) == ["CA", "NY"]
        assert list(result["city"]) == ["LA", "NYC"]

    def test_to_original_splits_columns_fallback(self):
        # test fallback: if only merged column exists, split it
        handler = FixedCombinationHandler(FixedCombination(columns=["state", "city"]))
        df = pd.DataFrame({"state|city": ["CA|LA", "NY|NYC"], "value": [1, 2]})

        result = handler.to_original(df)

        assert "state" in result.columns
        assert "city" in result.columns
        assert "state|city" not in result.columns
        assert list(result["state"]) == ["CA", "NY"]
        assert list(result["city"]) == ["LA", "NYC"]

    def test_round_trip(self):
        handler = FixedCombinationHandler(FixedCombination(columns=["a", "b", "c"]))
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"], "c": ["!", "@"], "other": [10, 20]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert set(df.columns) == set(restored.columns)
        for col in df.columns:
            assert list(df[col]) == list(restored[col])

    def test_encoding_types(self):
        handler = FixedCombinationHandler(FixedCombination(columns=["a", "b"]))
        assert handler.get_encoding_types() == {"a|b": "TABULAR_CATEGORICAL"}


class TestInequalityHandler:
    def test_to_internal_numeric(self):
        handler = InequalityHandler(Inequality(low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20, 30], "end": [15, 25, 35]})

        result = handler.to_internal(df)

        assert "start" in result.columns
        delta_col = [c for c in result.columns if c.startswith("__constraint_ineq_delta")][0]
        assert list(result[delta_col]) == [5, 5, 5]

    def test_to_internal_datetime(self):
        handler = InequalityHandler(Inequality(low_column="start", high_column="end"))
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
        handler = InequalityHandler(Inequality(low_column="start", high_column="end"))
        delta_col = handler._delta_column
        df = pd.DataFrame({"start": [10, 20], delta_col: [5, 10]})

        result = handler.to_original(df)

        assert "end" in result.columns
        assert list(result["end"]) == [15, 30]
        assert delta_col not in result.columns

    def test_to_original_datetime(self):
        handler = InequalityHandler(Inequality(low_column="start", high_column="end"))
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
        handler = InequalityHandler(Inequality(low_column="low", high_column="high"))
        df = pd.DataFrame({"low": [1.0, 2.0, 3.0], "high": [5.0, 7.0, 10.0], "other": ["a", "b", "c"]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["low"]) == [1.0, 2.0, 3.0]
        assert list(restored["high"]) == [5.0, 7.0, 10.0]

    def test_round_trip_datetime(self):
        handler = InequalityHandler(Inequality(low_column="start", high_column="end"))
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
        handler = InequalityHandler(Inequality(low_column="low", high_column="high"))
        df = pd.DataFrame({"low": [10, 20], "high": [5, 25]})  # first row violates

        internal = handler.to_internal(df)
        delta_col = handler._delta_column

        assert internal[delta_col].iloc[0] == 5  # corrected to abs
        assert internal[delta_col].iloc[1] == 5

    def test_encoding_types(self):
        handler = InequalityHandler(Inequality(low_column="a", high_column="b"))
        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_NUMERIC_AUTO"

    def test_strict_boundaries_false_allows_equality(self):
        """test that strict_boundaries=False (default) allows low == high."""
        handler = InequalityHandler(Inequality(low_column="start", high_column="end", strict_boundaries=False))
        df = pd.DataFrame({"start": [10, 20], "end": [10, 25]})
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] == 0  # equality allowed
        assert result[handler._delta_column].iloc[1] == 5

    def test_strict_boundaries_enforces_strict_inequality(self):
        """test that strict_boundaries=True enforces low < high for numeric and datetime."""
        # numeric
        handler = InequalityHandler(Inequality(low_column="start", high_column="end", strict_boundaries=True))
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
        handler = InequalityHandler(Inequality(low_column="start", high_column="end", strict_boundaries=True))
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
        handler = InequalityHandler(Inequality(low_column="low", high_column="high", strict_boundaries=True))

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
        handler = InequalityHandler(Inequality(low_column="start", high_column="end", strict_boundaries=True))

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


class TestRangeHandler:
    def test_to_internal_numeric(self):
        handler = RangeHandler(Range(low_column="min", middle_column="mid", high_column="max"))
        df = pd.DataFrame({"min": [0, 10], "mid": [5, 15], "max": [10, 20]})

        result = handler.to_internal(df)

        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        assert list(result[delta1_col]) == [5, 5]
        assert list(result[delta2_col]) == [5, 5]

    def test_to_internal_datetime(self):
        handler = RangeHandler(Range(low_column="start", middle_column="middle", high_column="end"))
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
        handler = RangeHandler(Range(low_column="min", middle_column="mid", high_column="max"))
        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        df = pd.DataFrame({"min": [0, 100], delta1_col: [5, 10], delta2_col: [5, 20]})

        result = handler.to_original(df)

        assert list(result["mid"]) == [5, 110]
        assert list(result["max"]) == [10, 130]

    def test_round_trip_numeric(self):
        handler = RangeHandler(Range(low_column="a", middle_column="b", high_column="c"))
        df = pd.DataFrame({"a": [0.0, 100.0], "b": [50.0, 150.0], "c": [100.0, 200.0]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["a"]) == [0.0, 100.0]
        assert list(restored["b"]) == [50.0, 150.0]
        assert list(restored["c"]) == [100.0, 200.0]

    def test_round_trip_datetime(self):
        handler = RangeHandler(Range(low_column="start", middle_column="middle", high_column="end"))
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
        handler = RangeHandler(Range(low_column="a", middle_column="b", high_column="c"))
        df = pd.DataFrame({"a": [10], "b": [5], "c": [20]})  # b < a violates

        internal = handler.to_internal(df)

        delta1_col = handler._delta1_column
        delta2_col = handler._delta2_column
        assert internal[delta1_col].iloc[0] == 5  # corrected to abs
        assert internal[delta2_col].iloc[0] == 15  # corrected to abs

    def test_encoding_types(self):
        handler = RangeHandler(Range(low_column="a", middle_column="b", high_column="c"))
        enc = handler.get_encoding_types()
        assert len(enc) == 2
        assert all(v == "TABULAR_NUMERIC_AUTO" for v in enc.values())


class TestOneHotEncodingHandler:
    def test_to_internal_converts_to_categorical(self):
        handler = OneHotEncodingHandler(OneHotEncoding(columns=["is_a", "is_b", "is_c"]))
        df = pd.DataFrame({"is_a": [1, 0, 0], "is_b": [0, 1, 0], "is_c": [0, 0, 1], "other": [10, 20, 30]})

        result = handler.to_internal(df)

        internal_col = handler._internal_column
        assert internal_col in result.columns
        assert list(result[internal_col]) == ["is_a", "is_b", "is_c"]
        # original columns are kept during to_internal
        assert "is_a" in result.columns

    def test_to_original_creates_onehot(self):
        handler = OneHotEncodingHandler(OneHotEncoding(columns=["cat_a", "cat_b", "cat_c"]))
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
        handler = OneHotEncodingHandler(OneHotEncoding(columns=["x", "y", "z"]))
        df = pd.DataFrame({"x": [1, 0, 0, 0], "y": [0, 1, 0, 0], "z": [0, 0, 1, 0], "value": [100, 200, 300, 400]})

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert list(restored["x"]) == [1, 0, 0, 0]
        assert list(restored["y"]) == [0, 1, 0, 0]
        assert list(restored["z"]) == [0, 0, 1, 0]

    def test_encoding_types(self):
        handler = OneHotEncodingHandler(OneHotEncoding(columns=["a", "b"]))
        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_CATEGORICAL"

    def test_handles_null_rows(self):
        handler = OneHotEncodingHandler(OneHotEncoding(columns=["col_a", "col_b"]))
        internal_col = handler._internal_column
        df = pd.DataFrame({internal_col: ["col_a", None, "col_b"], "other": [1, 2, 3]})

        result = handler.to_original(df)

        assert list(result["col_a"]) == [1, 0, 0]
        assert list(result["col_b"]) == [0, 0, 1]

    def test_handles_all_zeros_row(self):
        handler = OneHotEncodingHandler(OneHotEncoding(columns=["a", "b", "c"]))
        df = pd.DataFrame({"a": [1, 0], "b": [0, 0], "c": [0, 0], "other": [10, 20]})

        internal = handler.to_internal(df)

        internal_col = handler._internal_column
        assert internal[internal_col].iloc[0] == "a"
        assert internal[internal_col].iloc[1] is None


class TestConstraintTranslator:
    def test_mixed_constraints(self):
        constraints = [
            FixedCombination(columns=["state", "city"]),
            Inequality(low_column="start_age", high_column="end_age"),
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
            FixedCombination(columns=["a", "b"]),
            Inequality(low_column="low", high_column="high"),
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
        constraints = [FixedCombination(columns=["a", "b"])]
        translator = ConstraintTranslator(constraints)
        internal = ["a|b", "other"]

        original = translator.get_original_columns(internal)

        assert original == ["a", "b", "other"]

    def test_get_encoding_types(self):
        constraints = [
            FixedCombination(columns=["a", "b"]),
            Range(low_column="x", middle_column="y", high_column="z"),
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
                    tabular_model_configuration=ModelConfiguration(
                        constraints=[FixedCombination(columns=["state", "city"])]
                    ),
                )
            ],
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
                    language_model_configuration=ModelConfiguration(
                        constraints=[FixedCombination(columns=["country", "language"])]
                    ),
                )
            ],
        )

        translator, original_columns = ConstraintTranslator.from_generator_config(
            generator=generator, table_name="docs"
        )

        assert translator is not None
        assert original_columns == ["country", "language"]


class TestDomainValidation:
    def test_fixed_combination_requires_two_columns(self):
        with pytest.raises(ValueError, match="at least 2 columns"):
            FixedCombination(columns=["single"])

    def test_inequality_same_column_fails(self):
        with pytest.raises(ValueError, match="must be different"):
            Inequality(low_column="col", high_column="col")

    def test_range_duplicate_columns_fails(self):
        with pytest.raises(ValueError, match="must all be different"):
            Range(low_column="a", middle_column="a", high_column="b")

        with pytest.raises(ValueError, match="must all be different"):
            Range(low_column="a", middle_column="b", high_column="a")

    def test_onehot_requires_two_columns(self):
        with pytest.raises(ValueError, match="at least 2 columns"):
            OneHotEncoding(columns=["single"])

    def test_valid_constraints_create(self):
        fc = FixedCombination(columns=["a", "b", "c"])
        assert fc.columns == ["a", "b", "c"]

        ineq = Inequality(low_column="start", high_column="end")
        assert ineq.low_column == "start"

        rng = Range(low_column="min", middle_column="mid", high_column="max")
        assert rng.middle_column == "mid"

        ohe = OneHotEncoding(columns=["is_x", "is_y"])
        assert ohe.columns == ["is_x", "is_y"]
