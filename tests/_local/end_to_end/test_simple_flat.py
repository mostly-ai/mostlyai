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

import os
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from mostlyai.sdk import MostlyAI
from mostlyai.sdk._data.metadata_objects import ColumnSchema, TableSchema
from mostlyai.sdk.client.exceptions import APIStatusError
from mostlyai.sdk.domain import GeneratorConfig, ModelEncodingType, ProgressStatus, SyntheticDatasetConfig

load_dotenv()

s3_config = {
    "access_key": os.getenv("E2E_CLIENT_S3_ACCESS_KEY", "test-access-key"),
    "secret_key": os.getenv("E2E_CLIENT_S3_SECRET_KEY", "test-secret-key"),
    "bucket": os.getenv("E2E_CLIENT_S3_BUCKET", "test-bucket"),
}


@pytest.fixture(scope="module")
def mostly(tmp_path_factory, request):
    if request.param == "client":
        if os.getenv("E2E_CLIENT_S3_BUCKET") is None:
            pytest.skip("Client mode test requires extra env vars")
        yield MostlyAI(quiet=True)
    else:
        yield MostlyAI(local=True, local_dir=tmp_path_factory.mktemp("mostlyai"), quiet=True)


@pytest.mark.parametrize(
    "mostly, encoding_types",
    [
        ("local", {"a": "AUTO", "b": "AUTO"}),
        ("local", {"a": "LANGUAGE_CATEGORICAL", "b": "LANGUAGE_NUMERIC"}),
        ("client", {"a": "AUTO", "b": "AUTO"}),
        ("client", {"a": "LANGUAGE_CATEGORICAL", "b": "LANGUAGE_NUMERIC"}),
    ],
    ids=[
        "AUTO encoding types (local mode)",
        "LANGUAGE-only encoding types (local mode)",
        "AUTO encoding types (client mode)",
        "LANGUAGE-only encoding types (client mode)",
    ],
    indirect=["mostly"],
)
def test_simple_flat(tmp_path, mostly, encoding_types):
    ## ===== GENERATOR =====

    # Test 1: create/delete
    df = pd.DataFrame(
        {
            "id": range(200),
            "a": ["a1", "a2"] * 100,
            "b": [1, 2] * 100,
            "text": ["c", "d"] * 100,
        }
    )
    df.to_csv(tmp_path / "test.csv", index=False)
    config = {
        "name": "SDK E2E Test 1",
        "tables": [
            {
                "name": "data",
                "data": df,
                "primary_key": "id",
                "columns": [
                    {"name": "id", "model_encoding_type": "AUTO"},
                    {"name": "a", "model_encoding_type": encoding_types["a"]},
                    {"name": "b", "model_encoding_type": encoding_types["b"]},
                ],
                "tabular_model_configuration": {"max_epochs": 0.1},
                "language_model_configuration": {"max_epochs": 0.1},
            }
        ],
    }
    kwargs_list = [
        {"data": df, "name": "SDK E2E Test 1"},  # syntactic sugar: pass a dataframe directly
        {"data": tmp_path / "test.csv", "name": "SDK E2E Test 1"},  # syntactic sugar: pass a file path
        {
            "config": {
                "name": "SDK E2E Test 1",
                "tables": [
                    {
                        "name": "data",
                        "data": "https://github.com/mostly-ai/public-demo-data/raw/dev/census/census10k.parquet",
                    }
                ],
            }
        },  # syntactic sugar: pass a url in config
        {"config": config},  # config via dict
        {"config": GeneratorConfig(**config)},  # config via class
    ]
    for i, kwargs in enumerate(kwargs_list):
        g = mostly.train(**kwargs, start=False)
        assert g.name == "SDK E2E Test 1"
        if i < len(kwargs_list) - 1:
            g.delete()

    # Test 2: update
    g.update(name="SDK E2E Test 2")
    assert g.name == "SDK E2E Test 2"
    g = mostly.generators.get(g.id)
    assert g.name == "SDK E2E Test 2"
    g_config = g.config()
    assert isinstance(g_config, GeneratorConfig)
    assert g_config.name == "SDK E2E Test 2"

    # train
    g.training.start()
    g.training.wait()
    assert g.training_status == "DONE"

    # Test 3: clone
    connector_cfg = {
        "name": "S3 Connector",
        "type": "S3_STORAGE",
        "access_type": "READ_PROTECTED",
        "config": {"accessKey": s3_config["access_key"]},
        "secrets": {"secretKey": s3_config["secret_key"]},
        "ssl": None,
    }
    connector = mostly.connect(config=connector_cfg, test_connection=False)
    # the creation of a generator requires actual connection to auto detect model encoding types
    # so we mock the response of location_schema as the S3 credentials are dummy in the local mode tests
    location_schema_patch = (
        patch(
            "mostlyai.sdk._local.connectors.fetch_location_schema",
            return_value=TableSchema(
                columns=[
                    ColumnSchema(name="col1", default_model_encoding_type=ModelEncodingType.tabular_numeric_auto.value),
                    ColumnSchema(name="col2", default_model_encoding_type=ModelEncodingType.tabular_categorical.value),
                ],
            ),
        )
        if mostly.local
        else nullcontext()
    )
    with location_schema_patch:
        new_g = mostly.generators.create(
            config={
                "name": "SDK E2E Test 3",
                "tables": [
                    {
                        "name": "test_table",
                        "source_connector_id": connector.id,
                        "location": f"{s3_config['bucket']}/automation/players_csv/bb_players.csv",
                    }
                ],
            }
        )
        new_g_clone = new_g.clone(training_status="NEW")
    assert new_g_clone.name == "Clone - SDK E2E Test 3"
    assert new_g_clone.tables[0].source_connector_id == connector.id
    assert new_g_clone.training_status == ProgressStatus.new
    new_g.delete()
    new_g_clone.delete()

    # export / import
    g.export_to_file(tmp_path / "generator.zip")
    g.delete()
    g = mostly.generators.import_from_file(tmp_path / "generator.zip")

    # cloning imported generator raises HTTPException for local due to no source connector ids
    with pytest.raises(APIStatusError):
        g.clone()

    # reports
    g.reports(tmp_path)

    # logs
    g.training.logs(tmp_path)

    ## ===== SYNTHETIC_PROBE =====

    df = mostly.probe(g, size=10)
    assert len(df) == 10

    df = mostly.probe(g, seed=pd.DataFrame({"a": ["a1"], "x": ["x"]}))
    assert len(df) == 1
    # assert df[["a", "x"]].values.tolist() == [["a1", "x"]] # TODO: activate this once staging is updated

    df = mostly.probe(g, seed=pd.DataFrame({"a": ["a1"] * 10}))
    assert len(df) == 10

    ## ===== SYNTHETIC_DATASET =====

    # config via sugar
    sd = mostly.generate(g, start=False)
    assert sd.tables[0].configuration.sample_size == 200
    sd.delete()

    # config via dict
    config = {"tables": [{"name": "data", "configuration": {"sample_size": 100}}]}
    sd = mostly.generate(g, config=config, start=False)
    assert sd.name == "SDK E2E Test 2"
    sd_config = sd.config()
    assert isinstance(sd_config, SyntheticDatasetConfig)
    assert sd_config.tables[0].configuration.sample_size == 100
    sd.delete()

    # config via class
    config = {"tables": [{"name": "data", "configuration": {"sample_size": 100}}]}
    config = SyntheticDatasetConfig(**config)
    sd = mostly.generate(g, config=config, start=False)

    # update
    sd.update(name="SDK E2E Test 2")
    assert sd.name == "SDK E2E Test 2"
    sd = mostly.synthetic_datasets.get(sd.id)
    assert sd.name == "SDK E2E Test 2"
    sd_config = sd.config()
    assert isinstance(sd_config, SyntheticDatasetConfig)
    assert sd_config.name == "SDK E2E Test 2"
    assert sd_config.tables[0].configuration.sample_size == 100

    # generate
    sd.generation.start()
    sd.generation.wait()
    assert sd.generation_status == "DONE"

    # download
    sd.download(tmp_path)
    syn = sd.data()
    assert len(syn) == 100
    assert list(syn.columns) == list(df.columns)

    # reports
    sd.reports(tmp_path)

    # logs
    sd.generation.logs(tmp_path)

    # final cleanup
    sd.delete()
    g.delete()


@pytest.mark.parametrize("mostly", ["local"], indirect=True)
def test_reproducibility(mostly):
    df = pd.DataFrame(
        {"a": np.random.choice(["a1", "a2", "a3", "a4"], size=150), "b": np.random.choice([1, 2, 3, 4], size=150)}
    )
    g_config = {
        "random_state": 42,
        "tables": [{"name": "data", "data": df, "tabular_model_configuration": {"max_epochs": 0.1}}],
    }
    g1 = mostly.train(config=g_config)
    g2 = mostly.train(config=g_config)
    assert g1.accuracy == g2.accuracy
    del g_config["random_state"]
    g3 = mostly.train(config=g_config)
    assert g1.accuracy != g3.accuracy
    sd_config = {"random_state": 43, "tables": [{"name": "data", "configuration": {"sample_size": 50}}]}
    sd1 = mostly.generate(g1, config=sd_config)
    sd2 = mostly.generate(g2, config=sd_config)
    assert sd1.data().equals(sd2.data())
    pr1 = mostly.probe(g1, config=sd_config)
    pr2 = mostly.probe(g2, config=sd_config)
    assert pr1.equals(pr2)
    del sd_config["random_state"]
    sd3 = mostly.generate(g1, config=sd_config)
    pr3 = mostly.probe(g1, config=sd_config)
    assert not sd1.data().equals(sd3.data())
    assert not pr1.equals(pr3)
