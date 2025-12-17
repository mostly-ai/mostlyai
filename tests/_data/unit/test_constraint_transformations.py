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
    GeneratorConfig,
    Inequality,
    ModelConfiguration,
    ModelEncodingType,
    SourceColumn,
    SourceColumnConfig,
    SourceTable,
    SourceTableConfig,
)


# Fixtures
@pytest.fixture
def sample_df():
    """basic dataframe for testing."""
    return pd.DataFrame(
        {"state": ["CA", "NY"], "city": ["LA", "NYC"], "start": [10, 20], "end": [15, 25], "value": [1, 2]}
    )


@pytest.fixture
def datetime_df():
    """dataframe with datetime columns."""
    return pd.DataFrame(
        {"start": pd.to_datetime(["2024-01-01", "2024-02-01"]), "end": pd.to_datetime(["2024-01-10", "2024-02-15"])}
    )


@pytest.fixture
def datetime_table():
    """table config with datetime encoding."""
    return SourceTable(
        name="test_table",
        columns=[
            SourceColumn(name="start", model_encoding_type=ModelEncodingType.tabular_datetime),
            SourceColumn(name="end", model_encoding_type=ModelEncodingType.tabular_datetime),
        ],
    )


@pytest.fixture
def fixed_combination_constraint():
    return FixedCombination(table_name="test_table", columns=["state", "city"])


@pytest.fixture
def inequality_constraint():
    return Inequality(table_name="test_table", low_column="start", high_column="end")


# FixedCombinationHandler Tests
class TestFixedCombinationHandler:
    def test_round_trip(self, sample_df, fixed_combination_constraint):
        """test complete round-trip transformation."""
        handler = FixedCombinationHandler(fixed_combination_constraint)
        internal = handler.to_internal(sample_df.copy())
        restored = handler.to_original(internal)

        assert set(sample_df.columns) == set(restored.columns)
        pd.testing.assert_frame_equal(sample_df, restored)

    def test_merged_column_created(self, sample_df, fixed_combination_constraint):
        """test that merged column is created and original columns preserved."""
        handler = FixedCombinationHandler(fixed_combination_constraint)
        result = handler.to_internal(sample_df.copy())

        assert handler.merged_name in result.columns
        assert "state" in result.columns
        assert "city" in result.columns

    def test_to_original_removes_merged_column(self, sample_df, fixed_combination_constraint):
        """test that to_original removes merged column."""
        handler = FixedCombinationHandler(fixed_combination_constraint)
        internal = handler.to_internal(sample_df.copy())
        result = handler.to_original(internal)

        assert handler.merged_name not in result.columns
        assert "state" in result.columns
        assert "city" in result.columns

    @pytest.mark.parametrize(
        "columns,data,expected",
        [
            (["state", "city"], {"state": ["CA"], "city": ["LA\x1eSF"]}, {"state": ["CA"], "city": ["LA\x1eSF"]}),
            (["a", "b"], {"a": ["x\x1ey"], "b": ["1|2"]}, {"a": ["x\x1ey"], "b": ["1|2"]}),
        ],
    )
    def test_separator_escaping(self, columns, data, expected):
        """test that separator characters are properly handled."""
        constraint = FixedCombination(table_name="test_table", columns=columns)
        handler = FixedCombinationHandler(constraint)
        df = pd.DataFrame(data)

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        for col, expected_values in expected.items():
            assert list(restored[col]) == expected_values

    def test_encoding_types(self, fixed_combination_constraint):
        """test encoding types for merged column."""
        handler = FixedCombinationHandler(fixed_combination_constraint)
        enc = handler.get_encoding_types()

        assert len(enc) == 1
        assert enc[handler.merged_name] == "TABULAR_CATEGORICAL"


# InequalityHandler Tests
class TestInequalityHandler:
    @pytest.mark.parametrize(
        "df_fixture,expected_delta_type",
        [
            ("datetime_df", pd.Timestamp),
        ],
    )
    def test_to_internal_creates_delta(self, df_fixture, expected_delta_type, datetime_table, request):
        """test that delta column is created with correct type."""
        df = request.getfixturevalue(df_fixture)
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="start", high_column="end"), table=datetime_table
        )

        result = handler.to_internal(df)
        delta_col = [c for c in result.columns if "TABULAR_CONSTRAINT_INEQ_DELTA" in c][0]

        assert delta_col in result.columns
        assert isinstance(result[delta_col].iloc[0], expected_delta_type)

    @pytest.mark.parametrize(
        "df_data",
        [
            {"low": [1.0, 2.0, 3.0], "high": [5.0, 7.0, 10.0]},
            {
                "start": pd.to_datetime(["2024-01-01", "2024-06-15"]),
                "end": pd.to_datetime(["2024-01-31", "2024-12-31"]),
            },
        ],
    )
    def test_round_trip(self, df_data):
        """test round-trip for numeric and datetime types."""
        low_col = "low" if "low" in df_data else "start"
        high_col = "high" if "high" in df_data else "end"

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

        pd.testing.assert_frame_equal(restored[[low_col, high_col]], df[[low_col, high_col]])

    def test_violation_correction(self):
        """test that violations (low > high) are corrected."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="low", high_column="high"))
        df = pd.DataFrame({"low": [10, 20], "high": [5, 25]})

        internal = handler.to_internal(df)

        assert internal[handler._delta_column].iloc[0] == 0  # violation corrected
        assert internal[handler._delta_column].iloc[1] == 5  # valid delta preserved

    @pytest.mark.parametrize(
        "strict,low_val,high_val,expect_positive_delta",
        [
            (False, 10, 10, False),  # equality allowed
            (True, 10, 10, True),  # equality corrected to positive delta
        ],
    )
    def test_strict_boundaries(self, strict, low_val, high_val, expect_positive_delta):
        """test strict_boundaries parameter."""
        handler = InequalityHandler(
            Inequality(table_name="test_table", low_column="low", high_column="high", strict_boundaries=strict)
        )
        df = pd.DataFrame({"low": [low_val], "high": [high_val]})

        result = handler.to_internal(df)
        delta = result[handler._delta_column].iloc[0]

        if expect_positive_delta:
            assert delta > 0
        else:
            assert delta == 0

    def test_missing_columns_raises_error(self):
        """test that missing columns raise clear error."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame({"start": [10, 20]})

        with pytest.raises(ValueError, match="Columns.*not found in dataframe"):
            handler.to_internal(df)

    def test_encoding_types(self):
        """test encoding types for delta column."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="a", high_column="b"))
        enc = handler.get_encoding_types()

        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_NUMERIC_AUTO"


# ConstraintTranslator Tests
class TestConstraintTranslator:
    def test_mixed_constraints_round_trip(self, sample_df):
        """test multiple constraint types together."""
        constraints = [
            FixedCombination(table_name="test_table", columns=["state", "city"]),
            Inequality(table_name="test_table", low_column="start", high_column="end"),
        ]
        translator = ConstraintTranslator(constraints)

        internal = translator.to_internal(sample_df.copy())
        restored = translator.to_original(internal)

        pd.testing.assert_frame_equal(sample_df, restored)

    def test_get_all_column_names(self):
        """test that all column names (original + generated) are returned."""
        constraints = [
            FixedCombination(table_name="test_table", columns=["a", "b"]),
            Inequality(table_name="test_table", low_column="low", high_column="high"),
        ]
        translator = ConstraintTranslator(constraints)
        original = ["a", "b", "low", "high", "other"]

        all_columns = translator.get_all_column_names(original)

        # check original columns present
        assert set(original).issubset(set(all_columns))
        # check generated columns present
        assert any("TABULAR_CONSTRAINT_FC" in c for c in all_columns)
        assert any("TABULAR_CONSTRAINT_INEQ_DELTA" in c for c in all_columns)

    @pytest.mark.parametrize(
        "has_constraints,expected_result",
        [
            (True, lambda t: t is not None),
            (False, lambda t: t is None),
        ],
    )
    def test_from_generator_config(self, has_constraints, expected_result):
        """test creation from generator config."""
        constraints = [FixedCombination(table_name="customers", columns=["state", "city"])] if has_constraints else []
        generator = Generator(
            id="test-gen",
            name="Test Generator",
            tables=[
                SourceTable(
                    name="customers",
                    columns=[
                        SourceColumn(name="state"),
                        SourceColumn(name="city"),
                    ],
                    tabular_model_configuration=ModelConfiguration(),
                )
            ],
            constraints=constraints,
        )

        translator = ConstraintTranslator.from_generator_config(generator=generator, table_name="customers")

        assert expected_result(translator)


# Domain Validation Tests
class TestDomainValidation:
    @pytest.mark.parametrize(
        "constraint_type,params,error_match",
        [
            (FixedCombination, {"table_name": "t", "columns": ["single"]}, "at least 2 columns"),
            (Inequality, {"table_name": "t", "low_column": "col", "high_column": "col"}, "must be different"),
        ],
    )
    def test_invalid_constraints(self, constraint_type, params, error_match):
        """test that invalid constraints raise appropriate errors."""
        with pytest.raises(ValueError, match=error_match):
            constraint_type(**params)

    @pytest.mark.parametrize(
        "constraints,columns,encoding_types,should_raise,match",
        [
            # overlapping columns
            (
                [
                    FixedCombination(table_name="t", columns=["col1", "col2"]),
                    FixedCombination(table_name="t", columns=["col2", "col3"]),
                ],
                [("col1", None), ("col2", None), ("col3", None)],
                [None, None, None],
                True,
                "referenced by multiple constraints",
            ),
            # wrong encoding type for FixedCombination
            (
                [FixedCombination(table_name="t", columns=["col1", "col2"])],
                [("col1", ModelEncodingType.tabular_numeric_auto), ("col2", ModelEncodingType.tabular_numeric_auto)],
                [ModelEncodingType.tabular_numeric_auto, ModelEncodingType.tabular_numeric_auto],
                True,
                "must have TABULAR_CATEGORICAL encoding type",
            ),
            # mixed numeric/datetime for Inequality
            (
                [Inequality(table_name="t", low_column="low", high_column="high")],
                [("low", ModelEncodingType.tabular_numeric_auto), ("high", ModelEncodingType.tabular_datetime)],
                [ModelEncodingType.tabular_numeric_auto, ModelEncodingType.tabular_datetime],
                True,
                "must both be either numeric or datetime",
            ),
            # valid case
            (
                [
                    FixedCombination(table_name="t", columns=["state", "city"]),
                    Inequality(table_name="t", low_column="start", high_column="end"),
                ],
                [
                    ("state", ModelEncodingType.tabular_categorical),
                    ("city", ModelEncodingType.tabular_categorical),
                    ("start", ModelEncodingType.tabular_numeric_auto),
                    ("end", ModelEncodingType.tabular_numeric_auto),
                ],
                [
                    ModelEncodingType.tabular_categorical,
                    ModelEncodingType.tabular_categorical,
                    ModelEncodingType.tabular_numeric_auto,
                    ModelEncodingType.tabular_numeric_auto,
                ],
                False,
                None,
            ),
        ],
    )
    def test_generator_config_validation(self, constraints, columns, encoding_types, should_raise, match):
        """test GeneratorConfig validation for various constraint scenarios."""
        table_columns = [
            SourceColumnConfig(name=name, model_encoding_type=enc_type)
            for (name, _), enc_type in zip(columns, encoding_types)
        ]
        tables = [SourceTableConfig(name="t", columns=table_columns)]

        if should_raise:
            with pytest.raises(ValueError, match=match):
                GeneratorConfig(constraints=constraints, tables=tables)
        else:
            config = GeneratorConfig(constraints=constraints, tables=tables)
            assert config is not None


# Edge Cases
class TestEdgeCases:
    @pytest.mark.parametrize(
        "df_data,description",
        [
            ({"start": pd.Series([], dtype=float), "end": pd.Series([], dtype=float)}, "empty dataframe"),
            ({"start": [10], "end": [20]}, "single row"),
            ({"start": [np.nan, np.nan], "end": [np.nan, np.nan]}, "all NaN values"),
        ],
    )
    def test_edge_cases(self, df_data, description):
        """test edge cases for InequalityHandler."""
        handler = InequalityHandler(Inequality(table_name="test_table", low_column="start", high_column="end"))
        df = pd.DataFrame(df_data)

        internal = handler.to_internal(df)
        restored = handler.to_original(internal)

        assert len(restored) == len(df)
        if description == "all NaN values":
            assert restored["end"].isna().all()
