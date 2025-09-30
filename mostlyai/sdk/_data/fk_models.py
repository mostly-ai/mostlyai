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

import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from mostlyai.engine._encoding_types.tabular.categorical import (
    analyze_categorical,
    analyze_reduce_categorical,
    encode_categorical,
)
from mostlyai.engine._encoding_types.tabular.numeric import analyze_numeric, analyze_reduce_numeric, encode_numeric


class EntityEncoder(nn.Module):
    def __init__(
        self,
        cardinalities: dict[str, int],
        sub_column_embedding_dim: int = 16,
        entity_hidden_dim: int = 16,
        entity_embedding_dim: int = 8,
    ):
        super().__init__()
        self.cardinalities = cardinalities
        self.sub_column_embedding_dim = sub_column_embedding_dim
        self.entity_hidden_dim = entity_hidden_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(num_embeddings=cardinality, embedding_dim=self.sub_column_embedding_dim)
                for col, cardinality in self.cardinalities.items()
            }
        )
        entity_dim = len(self.cardinalities) * self.sub_column_embedding_dim
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, self.entity_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.entity_hidden_dim, self.entity_embedding_dim),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = torch.cat([self.embeddings[col](inputs[col]) for col in inputs.keys()], dim=1)
        encoded = self.entity_encoder(embeddings)
        return encoded


class ParentChildMatcher(nn.Module):
    def __init__(
        self,
        parent_cardinalities: dict[str, int],
        child_cardinalities: dict[str, int],
        sub_column_embedding_dim: int = 16,
        entity_hidden_dim: int = 16,
        entity_embedding_dim: int = 8,
        similarity_hidden_dim: int = 16,
    ):
        super().__init__()
        self.entity_embedding_dim = entity_embedding_dim
        self.similarity_hidden_dim = similarity_hidden_dim
        self.parent_encoder = EntityEncoder(
            cardinalities=parent_cardinalities,
            sub_column_embedding_dim=sub_column_embedding_dim,
            entity_hidden_dim=entity_hidden_dim,
            entity_embedding_dim=self.entity_embedding_dim,
        )
        self.child_encoder = EntityEncoder(
            cardinalities=child_cardinalities,
            sub_column_embedding_dim=sub_column_embedding_dim,
            entity_hidden_dim=entity_hidden_dim,
            entity_embedding_dim=self.entity_embedding_dim,
        )
        self.similarity_layer = nn.Sequential(
            nn.Linear(3 * self.entity_embedding_dim, self.similarity_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.similarity_hidden_dim, 1),
        )

    def forward(self, parent_inputs: dict[str, torch.Tensor], child_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        parent_encoded = self.parent_encoder(parent_inputs)
        child_encoded = self.child_encoder(child_inputs)

        similarity = F.normalize(parent_encoded, dim=1) * F.normalize(child_encoded, dim=1)  # cosine similarity
        similarity_layer_input = torch.cat([parent_encoded, child_encoded, similarity], dim=1)
        probability_logit = self.similarity_layer(similarity_layer_input)
        probability = torch.sigmoid(probability_logit)

        return probability


def analyze_df(
    *,
    df: pd.DataFrame,
    primary_key: str | None = None,
    parent_key: str | None = None,
    data_columns: list[str] | None = None,
    pre_training_dir: Path,
) -> None:
    t0 = time.time()

    pre_training_dir.mkdir(parents=True, exist_ok=True)

    key_columns = []
    if primary_key is not None:
        key_columns.append(primary_key)
    if parent_key is not None:
        key_columns.append(parent_key)

    data_columns = data_columns or list(df.columns)
    data_columns = list(set(data_columns).difference(key_columns).intersection(df.columns))
    num_columns = list(df.select_dtypes(include="number").columns.intersection(data_columns))
    cat_columns = list(df.columns.difference(num_columns).intersection(data_columns))

    # store pre-training metadata as JSON
    stats = {
        "primary_key": primary_key,
        "parent_key": parent_key,
        "data_columns": data_columns,
        "cat_columns": cat_columns,
        "num_columns": num_columns,
        "columns": {},
    }
    for col in data_columns:
        values = df[col]
        root_keys = pd.Series(np.arange(len(values)), name="root_keys")
        if col in cat_columns:
            analyze, reduce = analyze_categorical, analyze_reduce_categorical
        elif col in num_columns:
            analyze, reduce = analyze_numeric, analyze_reduce_numeric
        else:
            raise ValueError(f"unknown column type: {col}")
        col_stats = analyze(values, root_keys)
        col_stats = reduce([col_stats])
        stats["columns"][col] = col_stats

    # store stats
    stats_path = pre_training_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=4))

    t1 = time.time()
    print(f"pre_training() | time: {t1 - t0:.2f}s")


def encode_df(
    *, df: pd.DataFrame, pre_training_dir: Path, include_primary_key: bool = True, include_parent_key: bool = True
) -> pd.DataFrame:
    t0 = time.time()

    # load stats
    stats_path = pre_training_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    primary_key = stats["primary_key"]
    parent_key = stats["parent_key"]
    cat_columns = stats["cat_columns"]
    num_columns = stats["num_columns"]

    # encode columns
    data = []
    for col, col_stats in stats["columns"].items():
        if col in cat_columns:
            encode = encode_categorical
        elif col in num_columns:
            encode = encode_numeric
        else:
            raise ValueError(f"unknown column type: {col}")
        values = df[col].copy()
        df_encoded = encode(values, col_stats)
        df_encoded = df_encoded.add_prefix(col + "_")
        data.append(df_encoded)

    # optionally include keys
    for key, include_key in [(primary_key, include_primary_key), (parent_key, include_parent_key)]:
        if key is not None and include_key:
            data.insert(0, df[key])

    data = pd.concat(data, axis=1)

    t1 = time.time()
    print(f"encode_df() | time: {t1 - t0:.2f}s")
    return data


def prepare_training_data(
    parent_encoded_data: pd.DataFrame,
    tgt_encoded_data: pd.DataFrame,
    parent_primary_key: str,
    tgt_parent_key: str,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepare training data for a parent-child matching model.
    For each child, one positive pair and one negative pair will be sampled.
    If sample_size is not provided, all children will be sampled.
    Negative pairs are sampled randomly.

    Args:
        df_parents_encoded: Encoded parent data
        df_children_encoded: Encoded child data
        parents_primary_key: Primary key of parents
        children_foreign_key: Foreign key of children
        sample_size: Number of children to sample.
    """

    t0 = time.time()
    if sample_size is None:
        sample_size = len(tgt_encoded_data)

    parent_keys = parent_encoded_data[parent_primary_key].to_numpy()
    parents_X = parent_encoded_data.drop(columns=[parent_primary_key]).to_numpy(dtype=np.float32)
    n_parents = parents_X.shape[0]
    parent_index_by_key = pd.Series(np.arange(n_parents), index=parent_keys)

    child_keys = tgt_encoded_data[tgt_parent_key].to_numpy()
    children_X = tgt_encoded_data.drop(columns=[tgt_parent_key]).to_numpy(dtype=np.float32)
    n_children = children_X.shape[0]

    # sample children without replacement
    sample_size = min(int(sample_size), n_children)
    rng = np.random.default_rng()
    sampled_child_indices = rng.choice(n_children, size=sample_size, replace=False)
    children_X = children_X[sampled_child_indices]
    child_keys = child_keys[sampled_child_indices]

    # map each sampled child to its true parent row index
    true_parent_pos = parent_index_by_key.loc[child_keys].to_numpy()
    if np.any(pd.isna(true_parent_pos)):
        raise ValueError("Some child foreign keys do not match any parent primary key")

    # positive pairs
    pos_parents = parents_X[true_parent_pos]
    pos_labels = np.ones(sample_size, dtype=np.float32)

    # negative pairs; resample any collisions with true parent indices
    neg_indices = rng.integers(0, n_parents, size=sample_size)
    mask = neg_indices == true_parent_pos
    while mask.any():
        neg_indices[mask] = rng.integers(0, n_parents, size=mask.sum())
        mask = neg_indices == true_parent_pos
    neg_parents = parents_X[neg_indices]
    neg_labels = np.zeros(sample_size, dtype=np.float32)

    # concatenate positives and negatives
    parent_vecs = np.vstack([pos_parents, neg_parents]).astype(np.float32, copy=False)
    child_vecs = np.vstack([children_X, children_X]).astype(np.float32, copy=False)
    labels_vec = np.concatenate([pos_labels, neg_labels]).astype(np.float32, copy=False)

    # convert to pandas
    parent_pd = pd.DataFrame(parent_vecs, columns=parent_encoded_data.drop(columns=[parent_primary_key]).columns)
    child_pd = pd.DataFrame(child_vecs, columns=tgt_encoded_data.drop(columns=[tgt_parent_key]).columns)
    labels_pd = pd.Series(labels_vec, name="labels")

    t1 = time.time()
    print(f"prepare_training_data() | time: {t1 - t0:.2f}s")
    return parent_pd, child_pd, labels_pd


def train(
    *,
    model: ParentChildMatcher,
    parent_pd: pd.DataFrame,
    child_pd: pd.DataFrame,
    labels: torch.Tensor,
    do_plot_losses: bool = True,
) -> None:
    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    max_epochs = 1000

    X_parent = torch.tensor(parent_pd.values, dtype=torch.int64)
    X_child = torch.tensor(child_pd.values, dtype=torch.int64)
    y = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_parent, X_child, y)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    train_losses, val_losses = [], []
    best_model_state = None

    for epoch in range(max_epochs):
        # training phase
        model.train()
        train_loss = 0
        for batch_parent, batch_child, batch_y in train_loader:
            batch_parent = {col: batch_parent[:, i] for i, col in enumerate(parent_pd.columns)}
            batch_child = {col: batch_child[:, i] for i, col in enumerate(child_pd.columns)}
            optimizer.zero_grad()
            pred = model(batch_parent, batch_child)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_y.size(0)
        train_loss /= train_size
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_parent, batch_child, batch_y in val_loader:
                batch_parent = {col: batch_parent[:, i] for i, col in enumerate(parent_pd.columns)}
                batch_child = {col: batch_child[:, i] for i, col in enumerate(child_pd.columns)}
                pred = model(batch_parent, batch_child)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_y.size(0)
        val_loss /= val_size
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # early stopping check
        if val_loss < best_val_loss - 1e-5:  # small delta to avoid float issues
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    assert best_model_state is not None
    model.load_state_dict(best_model_state)
    print("Best model restored (lowest validation loss).")

    if do_plot_losses:
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train/Validation Loss")
        plt.show()


def store_fk_model(*, model: ParentChildMatcher, tgt_parent_key: str, fk_models_workspace_dir: Path) -> None:
    fk_models_workspace_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "parent_encoder": {
            "cardinalities": model.parent_encoder.cardinalities,
            "sub_column_embedding_dim": model.parent_encoder.sub_column_embedding_dim,
            "entity_hidden_dim": model.parent_encoder.entity_hidden_dim,
            "entity_embedding_dim": model.parent_encoder.entity_embedding_dim,
        },
        "child_encoder": {
            "cardinalities": model.child_encoder.cardinalities,
            "sub_column_embedding_dim": model.child_encoder.sub_column_embedding_dim,
            "entity_hidden_dim": model.child_encoder.entity_hidden_dim,
            "entity_embedding_dim": model.child_encoder.entity_embedding_dim,
        },
        "similarity_hidden_dim": model.similarity_hidden_dim,
    }
    model_config_path = fk_models_workspace_dir / f"model_config[{tgt_parent_key}].json"
    model_config_path.write_text(json.dumps(model_config, indent=4))
    model_state_path = fk_models_workspace_dir / f"model_weights[{tgt_parent_key}].pt"
    torch.save(model.state_dict(), model_state_path)


def load_fk_model(*, tgt_parent_key: str, fk_models_workspace_dir: Path) -> ParentChildMatcher:
    model_config_path = fk_models_workspace_dir / f"model_config[{tgt_parent_key}].json"
    model_config = json.loads(model_config_path.read_text())
    model = ParentChildMatcher(
        parent_cardinalities=model_config["parent_encoder"]["cardinalities"],
        child_cardinalities=model_config["child_encoder"]["cardinalities"],
        sub_column_embedding_dim=model_config["parent_encoder"]["sub_column_embedding_dim"],
        entity_hidden_dim=model_config["parent_encoder"]["entity_hidden_dim"],
        entity_embedding_dim=model_config["parent_encoder"]["entity_embedding_dim"],
        similarity_hidden_dim=model_config["similarity_hidden_dim"],
    )
    model_state_path = fk_models_workspace_dir / f"model_weights[{tgt_parent_key}].pt"
    model.load_state_dict(torch.load(model_state_path))
    return model


def build_parent_child_probabilities(
    *,
    model: ParentChildMatcher,
    tgt_encoded: pd.DataFrame,
    parent_encoded: pd.DataFrame,
) -> torch.Tensor:
    t0 = time.time()
    n_tgt = tgt_encoded.shape[0]
    n_parent = parent_encoded.shape[0]

    tgt_inputs = {col: torch.tensor(tgt_encoded[col].values.astype(np.int64)) for col in tgt_encoded.columns}
    parent_inputs = {col: torch.tensor(parent_encoded[col].values.astype(np.int64)) for col in parent_encoded.columns}

    # for all pairs, compute the probability for each (tgt, parent) pair
    tgt_inputs = {col: tgt_inputs[col].repeat_interleave(n_parent) for col in tgt_encoded.columns}
    parent_inputs = {col: parent_inputs[col].repeat(n_tgt) for col in parent_encoded.columns}

    model.eval()
    with torch.no_grad():
        probs = model(parent_inputs, tgt_inputs).squeeze()
        prob_matrix = probs.view(n_tgt, n_parent)
        t1 = time.time()
    print(f"build_parent_child_probabilities() | time: {t1 - t0:.2f}s")
    return prob_matrix


def sample_best_parents(
    *,
    prob_matrix: torch.Tensor,
) -> np.ndarray:
    best_parent_indices = torch.argmax(prob_matrix, dim=1).cpu().numpy()
    return best_parent_indices


def get_cardinalities(*, pre_training_dir: Path) -> dict[str, int]:
    stats_path = pre_training_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    cardinalities = {
        f"{column}_{sub_column}": cardinality
        for column, column_stats in stats["columns"].items()
        for sub_column, cardinality in column_stats["cardinalities"].items()
    }
    return cardinalities
