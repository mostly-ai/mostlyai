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

import pandas as pd

from mostlyai.sdk._data.constraint_transformations import ConstraintTranslator
from mostlyai.sdk.domain import FixedCombination, Generator, ModelConfiguration, SourceColumn, SourceTable


def test_to_internal_merges_columns():
    """test that to_internal merges columns correctly."""

    # create translator
    constraints = [FixedCombination(columns=["state", "city"])]
    translator = ConstraintTranslator(constraints)

    # create sample dataframe
    df = pd.DataFrame(
        {
            "state": ["CA", "NY", "TX"],
            "city": ["LA", "NYC", "Houston"],
            "value": [1, 2, 3],
        }
    )

    # apply transformation
    df_internal = translator.to_internal(df)

    # verify merged column exists
    assert "state|city" in df_internal.columns

    # verify original columns are removed
    assert "state" not in df_internal.columns
    assert "city" not in df_internal.columns

    # verify value column is preserved
    assert "value" in df_internal.columns

    # verify merged values are correct
    assert list(df_internal["state|city"]) == ["CA|LA", "NY|NYC", "TX|Houston"]


def test_to_original_splits_columns():
    """test that to_original splits merged columns correctly."""

    # create translator
    constraints = [FixedCombination(columns=["state", "city"])]
    translator = ConstraintTranslator(constraints)

    # create internal dataframe
    df_internal = pd.DataFrame(
        {
            "state|city": ["CA|LA", "NY|NYC", "TX|Houston"],
            "value": [1, 2, 3],
        }
    )

    # apply reverse transformation
    df_original = translator.to_original(df_internal)

    # verify original columns exist
    assert "state" in df_original.columns
    assert "city" in df_original.columns

    # verify merged column is removed
    assert "state|city" not in df_original.columns

    # verify value column is preserved
    assert "value" in df_original.columns

    # verify split values are correct
    assert list(df_original["state"]) == ["CA", "NY", "TX"]
    assert list(df_original["city"]) == ["LA", "NYC", "Houston"]


def test_round_trip_transformation():
    """test that round trip transformation preserves data."""

    # create translator
    constraints = [FixedCombination(columns=["state", "city"])]
    translator = ConstraintTranslator(constraints)

    # create original dataframe
    df_original = pd.DataFrame(
        {
            "state": ["CA", "NY", "TX"],
            "city": ["LA", "NYC", "Houston"],
            "value": [1, 2, 3],
        }
    )

    # apply forward and reverse transformation
    df_internal = translator.to_internal(df_original)
    df_reconstructed = translator.to_original(df_internal)

    # verify columns are the same (order might differ)
    assert set(df_original.columns) == set(df_reconstructed.columns)

    # verify data is the same
    for col in df_original.columns:
        assert list(df_original[col]) == list(df_reconstructed[col])


def test_from_generator_config_with_constraints():
    """test loading constraints from generator config."""
    # create generator with constraints
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
                    SourceColumn(name="amount"),
                ],
                tabular_model_configuration=ModelConfiguration(
                    constraints=[FixedCombination(columns=["state", "city"])]
                ),
            )
        ],
    )

    # load translator from generator config
    translator, original_columns = ConstraintTranslator.from_generator_config(
        generator=generator, table_name="customers"
    )

    # verify translator was created
    assert translator is not None
    assert len(translator.constraints) == 1
    assert translator.constraints[0].columns == ["state", "city"]

    # verify original columns were extracted
    assert original_columns == ["id", "state", "city", "amount"]

    # verify transformation works
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "state": ["CA", "NY", "TX"],
            "city": ["LA", "NYC", "Houston"],
            "amount": [100, 200, 300],
        }
    )

    df_internal = translator.to_internal(df)
    assert "state|city" in df_internal.columns
    assert "state" not in df_internal.columns
    assert "city" not in df_internal.columns

    df_restored = translator.to_original(df_internal)
    assert "state" in df_restored.columns
    assert "city" in df_restored.columns
    assert "state|city" not in df_restored.columns


def test_from_generator_config_no_constraints():
    """test loading from generator with no constraints."""
    generator = Generator(
        id="test-gen",
        name="Test Generator",
        tables=[
            SourceTable(
                name="simple_table",
                columns=[SourceColumn(name="col1"), SourceColumn(name="col2")],
            )
        ],
    )

    translator, columns = ConstraintTranslator.from_generator_config(generator=generator, table_name="simple_table")

    assert translator is None
    assert columns is None


def test_from_generator_config_table_not_found():
    """test loading from generator with non-existent table."""
    generator = Generator(
        id="test-gen",
        name="Test Generator",
        tables=[
            SourceTable(
                name="existing_table",
                columns=[SourceColumn(name="col1")],
            )
        ],
    )

    translator, columns = ConstraintTranslator.from_generator_config(
        generator=generator, table_name="nonexistent_table"
    )

    assert translator is None
    assert columns is None


def test_from_generator_config_language_model():
    """test loading constraints from language model configuration."""
    generator = Generator(
        id="test-gen",
        name="Test Generator",
        tables=[
            SourceTable(
                name="documents",
                columns=[
                    SourceColumn(name="id"),
                    SourceColumn(name="country"),
                    SourceColumn(name="language"),
                    SourceColumn(name="text"),
                ],
                language_model_configuration=ModelConfiguration(
                    constraints=[FixedCombination(columns=["country", "language"])]
                ),
            )
        ],
    )

    translator, original_columns = ConstraintTranslator.from_generator_config(
        generator=generator, table_name="documents"
    )

    assert translator is not None
    assert len(translator.constraints) == 1
    assert translator.constraints[0].columns == ["country", "language"]
    assert original_columns == ["id", "country", "language", "text"]
