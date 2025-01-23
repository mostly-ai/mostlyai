# Copyright 2024-2025 MOSTLY AI
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

from mostlyai.sdk.domain import SourceTableConfig, GeneratorConfig, SourceColumn, ModelEncodingType


def test_source_column():
    # test valid calls
    SourceColumn(**{"name": "col1"})
    # test invalid calls
    with pytest.raises(ValueError):
        # missing column name
        SourceColumn()


def test_source_table_config():
    cols = [{"name": "id"}, {"name": "col1"}, {"name": "col2"}]
    fk_ctx = {"column": "col1", "referenced_table": "tbl2", "is_context": True}
    fk_non = {"column": "col2", "referenced_table": "tbl2", "is_context": False}
    # test valid calls
    SourceTableConfig(**{"name": "tbl1"})
    SourceTableConfig(**{"name": "tbl1", "primary_key": "id"})
    SourceTableConfig(**{"name": "tbl1", "foreign_keys": [fk_ctx]})
    SourceTableConfig(**{"name": "tbl1", "columns": cols})
    SourceTableConfig(**{"name": "tbl1", "columns": cols, "primary_key": "id"})
    SourceTableConfig(**{"name": "tbl1", "columns": cols, "primary_key": "id", "foreign_keys": [fk_ctx]})
    SourceTableConfig(**{"name": "tbl1", "columns": cols, "tabular_model_configuration": {"max_epochs": 1}})
    # test invalid calls
    with pytest.raises(ValueError):  # missing table name
        SourceTableConfig()
    with pytest.raises(ValueError):  # non-unique column names
        SourceTableConfig(**{"name": "tbl1", "columns": cols + [{"name": "id"}]})
    with pytest.raises(ValueError):  # non-unique foreign keys
        SourceTableConfig(**{"name": "tbl1", "foreign_keys": [fk_non, fk_non]})
    with pytest.raises(ValueError):  # PK missing in columns
        SourceTableConfig(**{"name": "tbl1", "columns": cols, "primary_key": "XXX"})
    with pytest.raises(ValueError):  # FK missing in columns
        SourceTableConfig(**{"name": "tbl1", "columns": cols, "foreign_keys": [fk_ctx | {"column": "XXX"}]})
    with pytest.raises(ValueError):  # more than one context FK
        SourceTableConfig(**{"name": "tbl1", "columns": cols, "foreign_keys": [fk_ctx, fk_ctx | {"column": "col2"}]})
    with pytest.raises(ValueError):  # PK == FK
        SourceTableConfig(
            **{
                "name": "tbl1",
                "columns": [{"name": "id"}],
                "primary_key": "id",
                "foreign_keys": [fk_non | {"column": "id"}],
            }
        )
    with pytest.raises(ValueError):  # incorrectly provided tabular model configuration
        SourceTableConfig(
            **{
                "name": "tbl1",
                "columns": [{"name": "col1", "model_encoding_type": ModelEncodingType.language_text}],
                "tabular_model_configuration": {"max_epochs": 1},
            }
        )
    with pytest.raises(ValueError):  # incorrectly provided language model configuration
        SourceTableConfig(
            **{
                "name": "tbl1",
                "columns": [{"name": "col1", "model_encoding_type": ModelEncodingType.tabular_categorical}],
                "language_model_configuration": {"max_epochs": 1},
            }
        )


def test_generator_config():
    cols = [{"name": "id"}, {"name": "col1"}, {"name": "col2"}]
    # test valid calls
    GeneratorConfig()
    GeneratorConfig(**{"tables": [{"name": "tbl1"}]})
    GeneratorConfig(**{"tables": [{"name": "tbl1", "columns": cols}]})
    GeneratorConfig(**{"tables": [{"name": "tbl1", "primary_key": "id"}]})
    GeneratorConfig(
        **{
            "tables": [
                {
                    "name": "tbl1",
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
                    {
                        "name": "tbl1",
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
                    {
                        "name": "tbl1",
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
                    {
                        "name": "tbl1",
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
                    {
                        "name": "tbl1",
                        "columns": cols,
                        "primary_key": "id",
                        "foreign_keys": [{"column": "col1", "referenced_table": "t2", "is_context": True}],
                    },
                    {
                        "name": "t2",
                        "columns": cols,
                        "primary_key": "id",
                        "foreign_keys": [{"column": "col1", "referenced_table": "tbl1", "is_context": True}],
                    },
                ]
            }
        )
