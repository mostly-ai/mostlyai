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

import pytest

from mostlyai.sdk.domain import (
    Generator,
    GeneratorConfig,
    ModelEncodingType,
    SourceColumn,
    SourceTableConfig,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
    SyntheticTableConfiguration,
)


def test_source_column():
    # test valid calls
    SourceColumn(**{"name": "col1"})
    # test invalid calls
    with pytest.raises(ValueError):
        # missing column name
        SourceColumn()


def test_source_table_config():
    common_fields = {"name": "tbl1", "source_connector_id": "test_connector", "location": "path/to/data.parquet"}
    cols = [{"name": "id"}, {"name": "col1"}, {"name": "col2"}]
    fk_ctx = {"column": "col1", "referenced_table": "tbl2", "is_context": True}
    fk_non = {"column": "col2", "referenced_table": "tbl2", "is_context": False}
    # test valid calls
    SourceTableConfig(**common_fields)
    SourceTableConfig(**common_fields | {"primary_key": "id"})
    SourceTableConfig(**common_fields | {"foreign_keys": [fk_ctx]})
    SourceTableConfig(**common_fields | {"columns": cols})
    SourceTableConfig(**common_fields | {"columns": cols, "primary_key": "id"})
    SourceTableConfig(**common_fields | {"columns": cols, "primary_key": "id", "foreign_keys": [fk_ctx]})
    SourceTableConfig(**common_fields | {"columns": cols, "tabular_model_configuration": {"max_epochs": 1}})
    # test invalid calls
    with pytest.raises(ValueError):  # missing table name
        SourceTableConfig()
    with pytest.raises(ValueError):  # non-unique column names
        SourceTableConfig(**common_fields | {"columns": cols + [{"name": "id"}]})
    with pytest.raises(ValueError):  # non-unique foreign keys
        SourceTableConfig(**common_fields | {"foreign_keys": [fk_non, fk_non]})
    with pytest.raises(ValueError):  # PK missing in columns
        SourceTableConfig(**common_fields | {"columns": cols, "primary_key": "XXX"})
    with pytest.raises(ValueError):  # FK missing in columns
        SourceTableConfig(**common_fields | {"columns": cols, "foreign_keys": [fk_ctx | {"column": "XXX"}]})
    with pytest.raises(ValueError):  # more than one context FK
        SourceTableConfig(**common_fields | {"columns": cols, "foreign_keys": [fk_ctx, fk_ctx | {"column": "col2"}]})
    with pytest.raises(ValueError):  # PK == FK
        SourceTableConfig(
            **common_fields
            | {
                "columns": [{"name": "id"}],
                "primary_key": "id",
                "foreign_keys": [fk_non | {"column": "id"}],
            }
        )


def test_source_table_config_add_model_configuration():
    common_fields = {"name": "tbl1", "source_connector_id": "test_connector", "location": "path/to/data.parquet"}

    def assert_model_configuration(s: SourceTableConfig, has_tabular_model: bool, has_language_model: bool):
        assert (s.tabular_model_configuration is not None) is has_tabular_model
        assert (s.language_model_configuration is not None) is has_language_model

    # both model configurations will be present if SourceTableConfig.columns is None
    s = SourceTableConfig(**common_fields | {"primary_key": "id"})
    assert_model_configuration(s, has_tabular_model=True, has_language_model=True)
    # PK column only
    s = SourceTableConfig(**common_fields | {"primary_key": "id", "columns": [{"name": "id"}]})
    assert_model_configuration(s, has_tabular_model=True, has_language_model=False)

    # PK + language column (id: pk, name: lang_text)
    s = SourceTableConfig(
        **common_fields
        | {
            "primary_key": "id",
            "columns": [
                {"name": "id", "model_encoding_type": ModelEncodingType.tabular_categorical},
                {"name": "name", "model_encoding_type": ModelEncodingType.language_text},
            ],
        }
    )
    assert_model_configuration(s, has_tabular_model=True, has_language_model=True)

    # PK + FK columns
    s = SourceTableConfig(
        **common_fields
        | {
            "primary_key": "id",
            "foreign_keys": [{"column": "fk", "referenced_table": "tbl2", "is_context": True}],
            "columns": [{"name": "id"}, {"name": "fk"}],
        }
    )
    assert_model_configuration(s, has_tabular_model=True, has_language_model=False)

    # tabular column only
    s = SourceTableConfig(
        **common_fields | {"columns": [{"name": "col", "model_encoding_type": ModelEncodingType.tabular_categorical}]}
    )
    assert_model_configuration(s, has_tabular_model=True, has_language_model=False)

    # language column only
    s = SourceTableConfig(
        **common_fields | {"columns": [{"name": "col", "model_encoding_type": ModelEncodingType.language_text}]}
    )
    assert_model_configuration(s, has_tabular_model=False, has_language_model=True)

    # tabular and language columns
    s = SourceTableConfig(
        **common_fields
        | {
            "columns": [
                {"name": "col", "model_encoding_type": ModelEncodingType.tabular_categorical},
                {"name": "col2", "model_encoding_type": ModelEncodingType.language_text},
            ],
        }
    )
    assert_model_configuration(s, has_tabular_model=True, has_language_model=True)

    # language column with tabular model configuration
    s = SourceTableConfig(
        **common_fields
        | {
            "columns": [{"name": "col1", "model_encoding_type": ModelEncodingType.language_text}],
            "tabular_model_configuration": {"max_epochs": 1},
        }
    )
    assert_model_configuration(s, has_tabular_model=False, has_language_model=True)

    # tabular column with language model configuration
    s = SourceTableConfig(
        **common_fields
        | {
            "columns": [{"name": "col1", "model_encoding_type": ModelEncodingType.tabular_categorical}],
            "language_model_configuration": {"max_epochs": 1},
        }
    )
    assert_model_configuration(s, has_tabular_model=True, has_language_model=False)


def test_generator_config():
    common_fields = {"name": "tbl1", "source_connector_id": "test_connector", "location": "path/to/data.parquet"}
    cols = [{"name": "id"}, {"name": "col1"}, {"name": "col2"}]
    # test valid calls
    GeneratorConfig()
    GeneratorConfig(**{"tables": [common_fields]})
    GeneratorConfig(**{"tables": [common_fields | {"columns": cols}]})
    GeneratorConfig(**{"tables": [common_fields | {"primary_key": "id"}]})
    GeneratorConfig(
        **{
            "tables": [
                common_fields
                | {
                    "columns": cols,
                    "primary_key": "id",
                    "foreign_keys": [{"column": "col1", "referenced_table": "tbl1", "is_context": False}],
                }
            ]
        }
    )
    # test invalid calls
    with pytest.raises(ValueError):  # non-unique table names
        GeneratorConfig(**{"tables": [{"name": "tbl1"}, {"name": "tbl1"}]})
    with pytest.raises(ValueError):  # missing referenced table
        GeneratorConfig(
            **{
                "tables": [
                    common_fields
                    | {
                        "columns": cols,
                        "foreign_keys": [{"column": "col1", "referenced_table": "XXX", "is_context": True}],
                    }
                ]
            }
        )
    with pytest.raises(ValueError):  # missing PK in referenced table
        GeneratorConfig(
            **{
                "tables": [
                    common_fields
                    | {
                        "columns": cols,
                        "foreign_keys": [{"column": "col1", "referenced_table": "tbl", "is_context": False}],
                    }
                ]
            }
        )

    with pytest.raises(ValueError):  # self-referential context reference
        GeneratorConfig(
            **{
                "tables": [
                    common_fields
                    | {
                        "columns": cols,
                        "primary_key": "id",
                        "foreign_keys": [{"column": "col1", "referenced_table": "tbl1", "is_context": True}],
                    }
                ]
            }
        )
    with pytest.raises(ValueError):  # circular context reference
        GeneratorConfig(
            **{
                "tables": [
                    common_fields
                    | {
                        "columns": cols,
                        "primary_key": "id",
                        "foreign_keys": [{"column": "col1", "referenced_table": "t2", "is_context": True}],
                    },
                    common_fields
                    | {
                        "name": "t2",
                        "columns": cols,
                        "primary_key": "id",
                        "foreign_keys": [{"column": "col1", "referenced_table": "tbl1", "is_context": True}],
                    },
                ]
            }
        )


def test_synthetic_dataset_config():
    # test valid calls
    SyntheticDatasetConfig()
    SyntheticDatasetConfig(**{"name": "test", "description": "test desc"})
    SyntheticDatasetConfig(**{"tables": [{"name": "tbl1"}]})

    # test invalid calls
    with pytest.raises(ValueError):  # non-unique table names
        SyntheticDatasetConfig(**{"tables": [{"name": "tbl1"}, {"name": "tbl1"}]})


@pytest.mark.parametrize("config_class", [SyntheticDatasetConfig, SyntheticProbeConfig])
def test_synthetic_dataset_config_validate_against_generator(config_class):
    # prepare test data
    generator_cols = [
        SourceColumn(**{"name": "id", "model_encoding_type": "TABULAR_CATEGORICAL"}),
        SourceColumn(**{"name": "col1", "model_encoding_type": "TABULAR_CATEGORICAL"}),
        SourceColumn(**{"name": "col2", "model_encoding_type": "LANGUAGE_TEXT"}),
    ]
    generator = Generator(
        **{
            "id": "gen1",
            "tables": [
                {
                    "name": "tbl1",
                    "columns": generator_cols,
                    "total_rows": 100,
                    "primary_key": "id",
                    "tabular_model_configuration": {},
                }
            ],
        }
    )

    # test valid calls
    config = config_class(
        **{
            "tables": [
                {
                    "name": "tbl1",
                    "configuration": {
                        "rebalancing": {"column": "col1", "probabilities": {}},
                        "imputation": {"columns": ["col1"]},
                        "fairness": {"target_column": "col1", "sensitive_columns": ["id"]},
                    },
                }
            ]
        }
    )
    config.validate_against_generator(generator)

    # test sample_size defaults
    expected_sample_size = 1 if config_class == SyntheticProbeConfig else 100
    config = config_class(
        **{"tables": [{"name": "tbl1", "configuration": SyntheticTableConfiguration(sample_size=None)}]}
    )
    config.validate_against_generator(generator)
    assert config.tables[0].configuration.sample_size == expected_sample_size

    config = config_class(
        **{"tables": [{"name": "tbl1", "configuration": SyntheticTableConfiguration(sample_size=50)}]}
    )
    config.validate_against_generator(generator)
    assert config.tables[0].configuration.sample_size == 50

    with pytest.raises(ValueError):  # extra table not in generator
        config_class(**{"tables": [{"name": "tbl1"}, {"name": "extra_table"}]}).validate_against_generator(generator)

    with pytest.raises(ValueError):  # rebalancing column not found
        config_class(
            **{"tables": [{"name": "tbl1", "configuration": {"rebalancing": {"column": "missing_col"}}}]}
        ).validate_against_generator(generator)

    with pytest.raises(ValueError):  # rebalancing on non-categorical column
        config_class(
            **{"tables": [{"name": "tbl1", "configuration": {"rebalancing": {"column": "col2"}}}]}
        ).validate_against_generator(generator)

    with pytest.raises(ValueError):  # imputation column not found
        config_class(
            **{"tables": [{"name": "tbl1", "configuration": {"imputation": {"columns": ["missing_col"]}}}]}
        ).validate_against_generator(generator)

    with pytest.raises(ValueError):  # fairness target column not found
        config_class(
            **{
                "tables": [
                    {
                        "name": "tbl1",
                        "configuration": {"fairness": {"target_column": "missing_col", "sensitive_columns": ["id"]}},
                    }
                ]
            }
        ).validate_against_generator(generator)

    with pytest.raises(ValueError):  # fairness sensitive column not found
        config_class(
            **{
                "tables": [
                    {
                        "name": "tbl1",
                        "configuration": {"fairness": {"target_column": "col1", "sensitive_columns": ["missing_col"]}},
                    }
                ]
            }
        ).validate_against_generator(generator)

    with pytest.raises(ValueError):  # target column cannot be sensitive column
        config_class(
            **{
                "tables": [
                    {
                        "name": "tbl1",
                        "configuration": {"fairness": {"target_column": "col1", "sensitive_columns": ["col1"]}},
                    }
                ]
            }
        ).validate_against_generator(generator)

    # test adding missing tables
    generator_with_multiple_tables = Generator(
        **{
            "id": "gen1",
            "tables": [
                {
                    "name": "tbl1",
                    "columns": generator_cols,
                    "total_rows": 100,
                    "primary_key": "id",
                    "tabular_model_configuration": {},
                },
                {
                    "name": "tbl2",
                    "columns": generator_cols,
                    "total_rows": 50,
                    "primary_key": "id",
                    "tabular_model_configuration": {},
                },
            ],
        }
    )

    # test that missing tables are added automatically
    config = config_class(**{"tables": [{"name": "tbl1"}]})
    config.validate_against_generator(generator_with_multiple_tables)
    assert len(config.tables) == 2
    assert {t.name for t in config.tables} == {"tbl1", "tbl2"}

    # test that empty tables list gets populated with all generator tables
    config = config_class(**{"tables": None})
    config.validate_against_generator(generator_with_multiple_tables)
    assert len(config.tables) == 2
    assert {t.name for t in config.tables} == {"tbl1", "tbl2"}
