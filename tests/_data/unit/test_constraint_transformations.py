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
)
from mostlyai.sdk.domain import (
    FixedCombination,
    Generator,
    Inequality,
    ModelConfiguration,
    ModelEncodingType,
    SourceColumn,
    SourceTable,
)


class TestFixedCombinationHandler:
    def test_to_internal_merges_columns(self):
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        df = pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"], "value": [1, 2]})

        result = handler.to_internal(df)

        assert handler.merged_name in result.columns
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
        assert handler.merged_name not in result.columns
        assert list(result["state"]) == ["CA", "NY"]
        assert list(result["city"]) == ["LA", "NYC"]

    def test_to_original_splits_columns_fallback(self):
        # test fallback: if only merged column exists, split it
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=["state", "city"]))
        # create internal format with record separator
        internal = handler.to_internal(pd.DataFrame({"state": ["CA", "NY"], "city": ["LA", "NYC"]}))
        df = pd.DataFrame(
            {col: internal[col] for col in [handler.merged_name, "state", "city"] if col in internal.columns}
        )
        # remove original columns to test fallback
        df = df.drop(columns=["state", "city"], errors="ignore")

        result = handler.to_original(df)

        assert "state" in result.columns
        assert "city" in result.columns
        assert handler.merged_name not in result.columns
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
        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert handler.merged_name in enc
        assert enc[handler.merged_name] == "TABULAR_CATEGORICAL"

    @pytest.mark.parametrize(
        "columns,df_data,expected",
        [
            (
                ["state", "city"],
                {"state": ["CA", "NY"], "city": ["LA\x1eSF", "NYC"], "value": [1, 2]},
                {"state": ["CA", "NY"], "city": ["LA\x1eSF", "NYC"]},
            ),
            (
                ["state", "city"],
                {"state": ["CA", "NY"], "city": ["LA|SF", "NYC"], "value": [1, 2]},
                {"state": ["CA", "NY"], "city": ["LA|SF", "NYC"]},
            ),
            (
                ["a", "b", "c"],
                {"a": ["x\x1ey", "z"], "b": ["1", "2\x1e3"], "c": ["!", "@"], "other": [10, 20]},
                {"a": ["x\x1ey", "z"], "b": ["1", "2\x1e3"], "c": ["!", "@"]},
            ),
        ],
    )
    def test_separator_escaping(self, columns, df_data, expected):
        """test that separator characters (record separator and pipe) in data are properly handled."""
        handler = FixedCombinationHandler(FixedCombination(table_name="test_table", columns=columns))
        df = pd.DataFrame(df_data)

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        for col, expected_values in expected.items():
            assert list(restored[col]) == expected_values


class TestInequalityHandler:
    @pytest.mark.parametrize(
        "df_data,expected_delta",
        [
            # (
            #     {"start": [10, 20, 30], "end": [15, 25, 35]},
            #     [5, 5, 5],
            # ),
            (
                {
                    "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "end": pd.to_datetime(["2024-01-10", "2024-02-15"]),
                },
                [pd.Timestamp("1970-01-10"), pd.Timestamp("1970-01-15")],  # datetime representation (epoch + timedelta)
            ),
        ],
    )
    def test_to_internal(self, df_data, expected_delta):
        """test to_internal for both numeric and datetime types."""
        # create table with datetime encoding types if testing datetime
        table = None
        if isinstance(expected_delta[0], pd.Timestamp):
            table = SourceTable(
                name="test_table",
                columns=[
                    SourceColumn(name="start", model_encoding_type=ModelEncodingType.tabular_datetime),
                    SourceColumn(name="end", model_encoding_type=ModelEncodingType.tabular_datetime),
                ],
            )
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end"), table=table
        )
        df = pd.DataFrame(df_data)

        result = handler.to_internal(df)

        assert "start" in result.columns
        delta_col = [c for c in result.columns if "TABULAR_CONSTRAINT_INEQ_DELTA" in c][0]
        # datetime deltas are now stored as datetime (epoch + timedelta)
        if isinstance(expected_delta[0], pd.Timestamp):
            pd.testing.assert_series_equal(result[delta_col], pd.Series(expected_delta), check_names=False)
        else:
            assert list(result[delta_col]) == expected_delta

    @pytest.mark.parametrize(
        "df_data,delta_values,expected_end",
        [
            # (
            #     {"start": [10, 20]},
            #     [5, 10],
            #     [15, 30],
            # ),
            (
                {"start": pd.to_datetime(["2024-01-01", "2024-02-01"])},
                [pd.Timestamp("1970-01-10"), pd.Timestamp("1970-01-15")],  # datetime representation (epoch + timedelta)
                [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-02-15")],
            ),
        ],
    )
    def test_to_original(self, df_data, delta_values, expected_end):
        """test to_original for both numeric and datetime types."""
        # create table with datetime encoding types if testing datetime
        table = None
        if isinstance(expected_end[0], pd.Timestamp):
            table = SourceTable(
                name="test_table",
                columns=[
                    SourceColumn(name="start", model_encoding_type=ModelEncodingType.tabular_datetime),
                    SourceColumn(name="end", model_encoding_type=ModelEncodingType.tabular_datetime),
                ],
            )
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end"), table=table
        )
        delta_col = handler._delta_column
        df = pd.DataFrame({**df_data, delta_col: delta_values})

        result = handler.to_original(df)

        assert "end" in result.columns
        assert delta_col not in result.columns
        if isinstance(expected_end[0], pd.Timestamp):
            assert result["end"].iloc[0] == expected_end[0]
            assert result["end"].iloc[1] == expected_end[1]
        else:
            assert list(result["end"]) == expected_end

    @pytest.mark.parametrize(
        "df_data",
        [
            {"low": [1.0, 2.0, 3.0], "high": [5.0, 7.0, 10.0], "other": ["a", "b", "c"]},
            {
                "start": pd.to_datetime(["2024-01-01", "2024-06-15"]),
                "end": pd.to_datetime(["2024-01-31", "2024-12-31"]),
            },
        ],
    )
    def test_round_trip(self, df_data):
        """test round-trip for both numeric and datetime types."""
        low_col = "low" if "low" in df_data else "start"
        high_col = "high" if "high" in df_data else "end"
        # create table with datetime encoding types if testing datetime
        table = None
        if isinstance(df_data[low_col][0], pd.Timestamp):
            table = SourceTable(
                name="test_table",
                columns=[
                    SourceColumn(name=low_col, model_encoding_type=ModelEncodingType.tabular_datetime),
                    SourceColumn(name=high_col, model_encoding_type=ModelEncodingType.tabular_datetime),
                ],
            )
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column=low_col, high_column=high_col), table=table
        )
        df = pd.DataFrame(df_data)

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        if isinstance(df[low_col].iloc[0], pd.Timestamp):
            pd.testing.assert_series_equal(restored[low_col], df[low_col])
            pd.testing.assert_series_equal(restored[high_col], df[high_col])
        else:
            assert list(restored[low_col]) == list(df[low_col])
            assert list(restored[high_col]) == list(df[high_col])

    def test_violation_correction(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="low", high_column="high"))
        df = pd.DataFrame({"low": [10, 20], "high": [5, 25]})  # first row violates

        internal = handler.to_internal(df)
        delta_col = handler._delta_column

        assert internal[delta_col].iloc[0] == 0
        assert internal[delta_col].iloc[1] == 5

    def test_encoding_types(self):
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="a", high_column="b"))
        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_NUMERIC_AUTO"

    def test_strict_boundaries_false_allows_equality(self):
        """test that strict_boundaries=False allows low == high."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end", strict_boundaries=False)
        )
        df = pd.DataFrame({"start": [10, 20], "end": [10, 25]})
        result = handler.to_internal(df)
        assert result[handler._delta_column].iloc[0] == 0
        assert result[handler._delta_column].iloc[1] == 5

    @pytest.mark.parametrize(
        "df_data",
        [
            {"start": [10, 20, 30], "end": [10, 25, 35]},
            {
                "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "end": pd.to_datetime(["2024-01-01", "2024-02-15"]),
            },
        ],
    )
    def test_strict_boundaries_enforces_inequality(self, df_data):
        """test that strict_boundaries=True enforces low < high."""
        # create table with datetime encoding types if testing datetime
        table = None
        if isinstance(df_data["start"][0], pd.Timestamp):
            table = SourceTable(
                name="test_table",
                columns=[
                    SourceColumn(name="start", model_encoding_type=ModelEncodingType.tabular_datetime),
                    SourceColumn(name="end", model_encoding_type=ModelEncodingType.tabular_datetime),
                ],
            )
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end", strict_boundaries=True),
            table=table,
        )
        df = pd.DataFrame(df_data)
        result = handler.to_internal(df)
        delta = result[handler._delta_column]
        # for datetime, delta is stored as datetime (epoch + timedelta), so compare with epoch
        # for numeric, delta is stored as numeric
        if pd.api.types.is_datetime64_any_dtype(delta):
            assert delta.iloc[0] > handler._DATETIME_EPOCH
        else:
            assert delta.iloc[0] > 0

    def test_strict_boundaries_round_trip(self):
        """test round-trip with strict_boundaries=True corrects equality."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="low", high_column="high", strict_boundaries=True)
        )
        df = pd.DataFrame({"low": [1.0, 2.0], "high": [1.0, 3.0]})
        internal = handler.to_internal(df)
        restored = handler.to_original(internal)
        assert all(restored["high"] > restored["low"])

    def test_missing_columns_raises_error(self):
        """test that missing columns raise a clear error message."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20]})  # missing "end" column

        with pytest.raises(ValueError, match="Columns.*not found in dataframe"):
            handler.to_internal(df)


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

        all_column_names = translator.get_all_column_names(original)

        # FixedCombination keeps all original columns + merged column
        assert "a" in all_column_names
        assert "b" in all_column_names
        fc_handler = translator.handlers[0]
        assert fc_handler.merged_name in all_column_names
        # Inequality keeps all original columns + delta column
        assert "high" in all_column_names
        assert "low" in all_column_names
        assert "other" in all_column_names
        assert any("TABULAR_CONSTRAINT_INEQ_DELTA" in c for c in all_column_names)

    def test_get_original_columns(self):
        constraints = [FixedCombination(table_name="test_table", columns=["a", "b"])]
        translator = ConstraintTranslator(constraints)
        fc_handler = translator.handlers[0]
        internal = [fc_handler.merged_name, "other"]

        original = translator.get_original_column_names(internal)

        assert original == ["a", "b", "other"]

    def test_get_encoding_types(self):
        constraints = [
            FixedCombination(table_name="test_table", columns=["a", "b"]),
            Inequality(table_name="test_table", low_column="x", high_column="y"),
        ]
        translator = ConstraintTranslator(constraints)

        enc = translator.get_encoding_types()

        fc_handler = translator.handlers[0]
        assert enc[fc_handler.merged_name] == "TABULAR_CATEGORICAL"
        assert sum(1 for v in enc.values() if v == "TABULAR_NUMERIC_AUTO") == 1

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


class TestDomainValidation:
    def test_fixed_combination_requires_two_columns(self):
        with pytest.raises(ValueError, match="at least 2 columns, got 1"):
            FixedCombination(table_name="test_table", columns=["single"])

    def test_inequality_same_column_fails(self):
        with pytest.raises(ValueError, match="must be different, both are 'col'"):
            Inequality(table_name="test_table", low_column="col", high_column="col")

    def test_valid_constraints_create(self):
        fc = FixedCombination(table_name="test_table", columns=["a", "b", "c"])
        assert fc.columns == ["a", "b", "c"]

        ineq = Inequality(table_name="test_table", low_column="start", high_column="end")
        assert ineq.low_column == "start"


class TestEdgeCases:
    """test edge cases and simple scenarios that may have been missed."""

    def test_empty_dataframe(self):
        """test that empty dataframes are handled gracefully."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": pd.Series([], dtype=float), "end": pd.Series([], dtype=float)})
        result = handler.to_internal(df)
        assert len(result) == 0
        restored = handler.to_original(result)
        assert len(restored) == 0

    def test_single_row(self):
        """test single row case."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10], "end": [20]})
        internal = handler.to_internal(df)
        restored = handler.to_original(internal)
        assert restored["start"].iloc[0] == 10
        assert restored["end"].iloc[0] == 20

    def test_all_nan_values(self):
        """test all NaN values handling."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [np.nan, np.nan], "end": [np.nan, np.nan]})
        internal = handler.to_internal(df)
        assert internal[handler._delta_column].isna().all()
        restored = handler.to_original(internal)
        assert restored["end"].isna().all()

    def test_multiple_constraints_same_table(self):
        """test multiple constraints on the same table work together."""
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
            }
        )

        internal = translator.to_internal(df)
        restored = translator.to_original(internal)

        # all values should be preserved
        assert list(restored["state"]) == ["CA", "NY"]
        assert list(restored["city"]) == ["LA", "NYC"]
        assert list(restored["start_age"]) == [20, 30]
        assert list(restored["end_age"]) == [25, 40]


class TestIntegration:
    """integration tests for multiple constraints interacting and advanced scenarios."""

    @pytest.mark.parametrize(
        "constraint,generator,should_raise,match",
        [
            (
                FixedCombination(table_name="customers", columns=["state", "city"]),
                Generator(
                    id="test-gen",
                    name="Test Generator",
                    tables=[
                        SourceTable(
                            name="customers",
                            columns=[SourceColumn(name="id"), SourceColumn(name="state"), SourceColumn(name="city")],
                        )
                    ],
                ),
                False,
                None,
            ),
            (
                FixedCombination(table_name="orders", columns=["state", "city"]),
                Generator(
                    id="test-gen",
                    name="Test Generator",
                    tables=[SourceTable(name="customers", columns=[SourceColumn(name="id")])],
                ),
                True,
                "table 'orders' referenced by constraint not found",
            ),
            (
                FixedCombination(table_name="customers", columns=["state", "city"]),
                Generator(
                    id="test-gen",
                    name="Test Generator",
                    tables=[
                        SourceTable(name="customers", columns=[SourceColumn(name="id"), SourceColumn(name="state")])
                    ],
                ),
                True,
                "column 'city' in table 'customers' referenced by constraint not found",
            ),
        ],
    )
    def test_validate_against_generator(self, constraint, generator, should_raise, match):
        """test constraint validation against generator config."""
        handler = FixedCombinationHandler(constraint)
        if should_raise:
            with pytest.raises(ValueError, match=match):
                handler.validate_against_generator(generator)
        else:
            handler.validate_against_generator(generator)
