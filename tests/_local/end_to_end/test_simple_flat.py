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

import numpy as np
from mostlyai.sdk.client.exceptions import APIStatusError
import pytest
from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import GeneratorConfig, SyntheticDatasetConfig, ProgressStatus
import pandas as pd


@pytest.mark.parametrize(
    "encoding_types",
    [{"a": "AUTO", "b": "AUTO"}, {"a": "LANGUAGE_CATEGORICAL", "b": "LANGUAGE_NUMERIC"}],
    ids=["AUTO encoding types", "LANGUAGE-only encoding types"],
)
def test_simple_flat(tmp_path, encoding_types):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    # create mock data
    df = pd.DataFrame(
        {
            "id": range(200),
            "a": ["a1", "a2"] * 100,
            "b": [1, 2] * 100,
            "text": ["c", "d"] * 100,
        }
    )

    ## GENERATOR

    # config via sugar
    g = mostly.train(data=df, name="Test 1", start=False)
    assert g.name == "Test 1"
    g.delete()

    # config via sugar
    df.to_csv(tmp_path / "test.csv", index=False)
    g = mostly.train(data=df, name="Test 1", start=False)
    assert g.name == "Test 1"
    assert len(df.columns) == len(g.tables[0].columns)
    g.delete()

    # config via dict
    config = {
        "name": "Test 1",
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
    g = mostly.train(config=config, start=False)
    assert g.name == "Test 1"
    g.delete()

    # config via dict and sugar
    g = mostly.train(
        config={
            "tables": [
                {
                    "name": "data",
                    "data": "https://github.com/mostly-ai/public-demo-data/raw/dev/census/census10k.parquet",
                }
            ]
        },
        start=False,
    )
    g.delete()

    # config via class
    g = mostly.train(config=GeneratorConfig(**config), start=False)
    assert g.name == "Test 1"

    # update
    g.update(name="Test 2")
    assert g.name == "Test 2"
    g = mostly.generators.get(g.id)
    assert g.name == "Test 2"
    g_config = g.config()
    assert isinstance(g_config, GeneratorConfig)
    assert g_config.name == "Test 2"

    # train
    g.training.start()
    g.training.wait()
    assert g.training_status == "DONE"

    # clone new generator
    connector_cfg = {
        "name": "Test 1",
        "type": "S3_STORAGE",
        "access_type": "SOURCE",
        "config": {"accessKey": "XXX"},
        "secrets": {"secretKey": "a89fb747734f4162bf36c8f1e853355f2176b583013c26e83c3324e453fd2e7b"},
        "ssl": None,
    }
    connector = mostly.connect(config=connector_cfg, test_connection=False)
    new_g = mostly.generators.create(
        config={
            "name": "Test 3",
            "tables": [
                {
                    "name": "test_table",
                    "source_connector_id": connector.id,
                }
            ],
        }
    )
    new_g_clone = new_g.clone(training_status="NEW")
    assert new_g_clone.name == "Test 3"
    assert new_g_clone.tables[0].source_connector_id == connector.id
    assert new_g_clone.training_status == ProgressStatus.new

    # export / import
    g.export_to_file(tmp_path / "generator.zip")
    g = mostly.generators.import_from_file(tmp_path / "generator.zip")

    # cloning imported generator raises HTTPException for local due to no source connector ids
    with pytest.raises(APIStatusError):
        g.clone()

    # reports
    g.reports(tmp_path)

    # logs
    g.training.logs(tmp_path)

    ## SYNTHETIC PROBE
    df = mostly.probe(g, size=10)
    assert len(df) == 10

    df = mostly.probe(g, seed=pd.DataFrame({"a": ["a1"] * 10}))
    assert len(df) == 10

    ## SYNTHETIC DATASET

    # config via sugar
    sd = mostly.generate(g, start=False)
    assert sd.tables[0].configuration.sample_size == 200
    sd.delete()

    # config via dict
    config = {"tables": [{"name": "data", "configuration": {"sample_size": 100}}]}
    sd = mostly.generate(g, config=config, start=False)
    assert sd.name == "Test 2"
    sd_config = sd.config()
    assert isinstance(sd_config, SyntheticDatasetConfig)
    assert sd_config.tables[0].configuration.sample_size == 100
    sd.delete()

    # config via class
    config = {"tables": [{"name": "data", "configuration": {"sample_size": 100}}]}
    config = SyntheticDatasetConfig(**config)
    sd = mostly.generate(g, config=config, start=False)

    # update
    sd.update(name="Test 2")
    assert sd.name == "Test 2"
    sd = mostly.synthetic_datasets.get(sd.id)
    assert sd.name == "Test 2"
    sd_config = sd.config()
    assert isinstance(sd_config, SyntheticDatasetConfig)
    assert sd_config.name == "Test 2"
    assert sd_config.tables[0].configuration.sample_size == 100

    # generate
    sd.generation.start()
    sd.generation.wait()
    assert sd.generation_status == "DONE"
    sd.download(tmp_path)
    syn = sd.data()
    assert len(syn) == 100
    assert list(syn.columns) == list(df.columns)

    # reports
    sd.reports(tmp_path)

    # logs
    sd.generation.logs(tmp_path)


def test_reproducibility(tmp_path):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)
    df = pd.DataFrame(
        {"a": np.random.choice(["a1", "a2", "a3", "a4"], size=150), "b": np.random.choice([1, 2, 3, 4], size=150)}
    )
    g_config = {
        "random_state": 42,
        "tables": [{"name": "data", "data": df, "model_configuration": {"max_epochs": 0.1}}],
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
    del sd_config["random_state"]
    sd3 = mostly.generate(g1, config=sd_config)
    assert not sd1.data().equals(sd3.data())
    pr1 = mostly.probe(g1, config=sd_config)
    pr2 = mostly.probe(g2, config=sd_config)
    assert pr1.equals(pr2)
