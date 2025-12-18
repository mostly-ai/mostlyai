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


@pytest.fixture
def df():
    """sample dataframe with categorical, numeric, and other columns."""
    return pd.DataFrame(
        {
            "a": ["x", "y"],
            "b": ["1", "2"],
            "low": [10, 20],
            "high": [15, 25],
            "other": [1, 2],
        }
    )


class TestFixedCombinationHandler:
    def test_round_trip(self, df):
        """test transformation preserves data."""
        constraint = FixedCombination(table_name="t", columns=["a", "b"])
        handler = FixedCombinationHandler(constraint)

        internal = handler.to_internal(df.copy())
        restored = handler.to_original(internal)

        pd.testing.assert_frame_equal(df, restored)

    def test_creates_merged_column(self, df):
        """test merged column is created and removed correctly."""
        constraint = FixedCombination(table_name="t", columns=["a", "b"])
        handler = FixedCombinationHandler(constraint)

        internal = handler.to_internal(df.copy())
        assert handler.merged_name in internal.columns
        assert "a" in internal.columns
        assert "b" in internal.columns

        restored = handler.to_original(internal)
        assert handler.merged_name not in restored.columns

    def test_encoding_type(self):
        """test encoding type is categorical."""
        constraint = FixedCombination(table_name="t", columns=["a", "b"])
        handler = FixedCombinationHandler(constraint)

        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_CATEGORICAL"

    @pytest.mark.parametrize(
        "a_val,b_val",
        [
            ("x\x1ey", "1|2"),
            ("[cat]", "val"),
            ("a'b", 'x"y'),
            ("a\tb", "x\ny"),
            ("{key}", "<val>"),
        ],
    )
    def test_escaping(self, a_val, b_val):
        """test special characters are handled."""
        constraint = FixedCombination(table_name="t", columns=["a", "b"])
        handler = FixedCombinationHandler(constraint)
        df = pd.DataFrame({"a": [a_val], "b": [b_val]})

        internal = handler.to_internal(df.copy())
        restored = handler.to_original(internal)

        assert restored["a"].iloc[0] == a_val
        assert restored["b"].iloc[0] == b_val


class TestInequalityHandler:
    def test_round_trip_numeric(self, df):
        """test numeric inequality round-trip."""
        constraint = Inequality(table_name="t", low_column="low", high_column="high")
        handler = InequalityHandler(constraint)

        internal = handler.to_internal(df.copy())
        restored = handler.to_original(internal)

        pd.testing.assert_frame_equal(df[["low", "high"]], restored[["low", "high"]])

    def test_round_trip_datetime(self):
        """test datetime inequality round-trip."""
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-06-15"]),
                "end": pd.to_datetime(["2024-01-31", "2024-12-31"]),
            }
        )
        table = SourceTable(
            name="t",
            columns=[
                SourceColumn(name="start", model_encoding_type=ModelEncodingType.tabular_datetime),
                SourceColumn(name="end", model_encoding_type=ModelEncodingType.tabular_datetime),
            ],
        )
        constraint = Inequality(table_name="t", low_column="start", high_column="end")
        handler = InequalityHandler(constraint, table=table)

        internal = handler.to_internal(df.copy())
        restored = handler.to_original(internal)

        pd.testing.assert_frame_equal(df, restored)

    def test_creates_delta_column(self):
        """test delta column is created."""
        df = pd.DataFrame({"low": [10], "high": [20]})
        constraint = Inequality(table_name="t", low_column="low", high_column="high")
        handler = InequalityHandler(constraint)

        internal = handler.to_internal(df.copy())
        delta_col = handler._delta_column

        assert delta_col in internal.columns
        assert internal[delta_col].iloc[0] == 10

    def test_violation_correction(self):
        """test violations are corrected."""
        df = pd.DataFrame({"low": [10, 20], "high": [5, 25]})
        constraint = Inequality(table_name="t", low_column="low", high_column="high")
        handler = InequalityHandler(constraint)

        internal = handler.to_internal(df.copy())

        assert internal[handler._delta_column].iloc[0] == 0
        assert internal[handler._delta_column].iloc[1] == 5

    def test_missing_columns_error(self):
        """test missing columns raise error."""
        df = pd.DataFrame({"low": [10]})
        constraint = Inequality(table_name="t", low_column="low", high_column="high")
        handler = InequalityHandler(constraint)

        with pytest.raises(ValueError, match="Columns.*not found"):
            handler.to_internal(df.copy())

    def test_encoding_type(self):
        """test encoding type is numeric."""
        constraint = Inequality(table_name="t", low_column="low", high_column="high")
        handler = InequalityHandler(constraint)

        enc = handler.get_encoding_types()
        assert len(enc) == 1
        assert list(enc.values())[0] == "TABULAR_NUMERIC_AUTO"

    @pytest.mark.parametrize(
        "data,desc",
        [
            ({"low": pd.Series([], dtype=float), "high": pd.Series([], dtype=float)}, "empty"),
            ({"low": [10], "high": [20]}, "single row"),
            ({"low": [np.nan], "high": [np.nan]}, "nan"),
        ],
    )
    def test_edge_cases(self, data, desc):
        """test edge cases."""
        constraint = Inequality(table_name="t", low_column="low", high_column="high")
        handler = InequalityHandler(constraint)
        df = pd.DataFrame(data)

        internal = handler.to_internal(df.copy())
        restored = handler.to_original(internal)

        assert len(restored) == len(df)
        if desc == "nan":
            assert restored["high"].isna().all()


class TestConstraintTranslator:
    def test_mixed_constraints(self, df):
        """test multiple constraint types together."""
        constraints = [
            FixedCombination(table_name="t", columns=["a", "b"]),
            Inequality(table_name="t", low_column="low", high_column="high"),
        ]
        translator = ConstraintTranslator(constraints)

        internal = translator.to_internal(df.copy())
        restored = translator.to_original(internal)

        pd.testing.assert_frame_equal(df, restored)

    def test_get_all_column_names(self):
        """test all column names are returned."""
        constraints = [
            FixedCombination(table_name="t", columns=["a", "b"]),
            Inequality(table_name="t", low_column="low", high_column="high"),
        ]
        translator = ConstraintTranslator(constraints)
        original = ["a", "b", "low", "high", "other"]

        all_cols = translator.get_all_column_names(original)

        assert set(original).issubset(set(all_cols))
        assert any("FC" in c for c in all_cols)
        assert any("DELTA" in c for c in all_cols)

    @pytest.mark.parametrize("has_constraints", [True, False])
    def test_from_generator_config(self, has_constraints):
        """test creation from generator config."""
        constraints = [FixedCombination(table_name="t", columns=["a", "b"])] if has_constraints else []
        generator = Generator(
            id="g",
            name="G",
            tables=[
                SourceTable(
                    name="t",
                    columns=[SourceColumn(name="a"), SourceColumn(name="b")],
                    tabular_model_configuration=ModelConfiguration(),
                )
            ],
            constraints=constraints,
        )

        translator = ConstraintTranslator.from_generator_config(generator=generator, table_name="t")

        assert (translator is not None) == has_constraints


class TestValidation:
    @pytest.mark.parametrize(
        "constraint_type,params,error_match",
        [
            (FixedCombination, {"table_name": "t", "columns": ["a"]}, "at least 2 columns"),
            (Inequality, {"table_name": "t", "low_column": "a", "high_column": "a"}, "must be different"),
        ],
    )
    def test_invalid_constraints(self, constraint_type, params, error_match):
        """test invalid constraints raise errors."""
        with pytest.raises(ValueError, match=error_match):
            constraint_type(**params)

    @pytest.mark.parametrize(
        "constraints,cols,types,should_fail,match",
        [
            (
                [
                    FixedCombination(table_name="t", columns=["a", "b"]),
                    FixedCombination(table_name="t", columns=["b", "c"]),
                ],
                ["a", "b", "c"],
                [None, None, None],
                True,
                "multiple constraints",
            ),
            (
                [FixedCombination(table_name="t", columns=["a", "b"])],
                ["a", "b"],
                [ModelEncodingType.tabular_numeric_auto, ModelEncodingType.tabular_numeric_auto],
                True,
                "CATEGORICAL",
            ),
            (
                [Inequality(table_name="t", low_column="low", high_column="high")],
                ["low", "high"],
                [ModelEncodingType.tabular_numeric_auto, ModelEncodingType.tabular_datetime],
                True,
                "numeric or datetime",
            ),
            (
                [
                    FixedCombination(table_name="t", columns=["a", "b"]),
                    Inequality(table_name="t", low_column="low", high_column="high"),
                ],
                ["a", "b", "low", "high"],
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
    def test_config_validation(self, constraints, cols, types, should_fail, match):
        """test generator config validation."""
        table_cols = [SourceColumnConfig(name=name, model_encoding_type=t) for name, t in zip(cols, types)]
        tables = [SourceTableConfig(name="t", columns=table_cols)]

        if should_fail:
            with pytest.raises(ValueError, match=match):
                GeneratorConfig(constraints=constraints, tables=tables)
        else:
            assert GeneratorConfig(constraints=constraints, tables=tables) is not None
