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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class ParentChildMatcher(nn.Module):
    def __init__(self, parent_dim: int, child_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.parent_dim = parent_dim
        self.child_dim = child_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.child_encoder = nn.Sequential(nn.Linear(child_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
        self.parent_encoder = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim)
        )
        self.similarity_layer = nn.Sequential(nn.Linear(3 * emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, parent_vec: torch.Tensor, child_vec: torch.Tensor) -> torch.Tensor:
        u = self.parent_encoder(parent_vec)
        v = self.child_encoder(child_vec)
        sim = F.normalize(u, dim=1) * F.normalize(v, dim=1)  # cosine similarity
        x = torch.cat([u, v, sim], dim=1)
        logit = self.similarity_layer(x)
        return torch.sigmoid(logit)


def pre_training(
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
    num_columns = df.select_dtypes(include="number").columns.intersection(data_columns)
    cat_columns = df.columns.difference(num_columns).intersection(data_columns)

    # fit encoder for numeric columns & store
    num_encoder = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    num_encoder.fit(df[num_columns])
    num_encoder_path = pre_training_dir / "num_encoder.pkl"
    joblib.dump(num_encoder, num_encoder_path)

    # fit encoder for categorical columns & store
    cat_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        max_categories=20,
        sparse_output=False,
    )
    df_cat = df[cat_columns].astype(str).fillna("__NULL__")
    cat_encoder = cat_encoder.fit(df_cat)
    cat_encoder_path = pre_training_dir / "cat_encoder.pkl"
    joblib.dump(cat_encoder, cat_encoder_path)

    # store pre-training metadata as JSON
    pre_training_meta = {
        "primary_key": primary_key,
        "parent_key": parent_key,
        "data_columns": data_columns,
    }
    pre_training_meta_path = pre_training_dir / "pre_training_meta.json"
    pre_training_meta_path.write_text(json.dumps(pre_training_meta, indent=4))

    t1 = time.time()
    print(f"pre_training() | time: {t1 - t0:.2f}s")


def encode_df(
    *, df: pd.DataFrame, pre_training_dir: Path, include_primary_key: bool = True, include_parent_key: bool = True
) -> pd.DataFrame:
    t0 = time.time()

    pre_training_meta_path = pre_training_dir / "pre_training_meta.json"
    pre_training_meta = json.loads(pre_training_meta_path.read_text())
    primary_key = pre_training_meta["primary_key"]
    parent_key = pre_training_meta["parent_key"]

    # encode numeric columns
    num_encoder_path = pre_training_dir / "num_encoder.pkl"
    num_encoder = joblib.load(num_encoder_path)
    num_encoded = num_encoder.transform(df[num_encoder.feature_names_in_])
    num_features = num_encoder.get_feature_names_out().tolist()

    # encode categorical columns
    cat_encoder_path = pre_training_dir / "cat_encoder.pkl"
    cat_encoder = joblib.load(cat_encoder_path)
    cat_df = df[cat_encoder.feature_names_in_].astype(str).fillna("__NULL__")
    cat_encoded = cat_encoder.transform(cat_df)
    cat_features = cat_encoder.get_feature_names_out().tolist()

    # concatenate keys and encoded columns
    data = [num_encoded, cat_encoded]
    features = num_features + cat_features
    for key, include_key in [(primary_key, include_primary_key), (parent_key, include_parent_key)]:
        if key is not None and include_key:
            data.insert(0, df[[key]])
            features.insert(0, key)
    data = np.concatenate(data, axis=1)

    t1 = time.time()
    print(f"encode_df() | time: {t1 - t0:.2f}s")
    return pd.DataFrame(data, columns=features)


def prepare_training_data(
    df_parents_encoded: pd.DataFrame,
    df_children_encoded: pd.DataFrame,
    parent_primary_key: str,
    children_foreign_key: str,
    sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        sample_size = len(df_children_encoded)

    parent_keys = df_parents_encoded[parent_primary_key].to_numpy()
    parents_X = df_parents_encoded.drop(columns=[parent_primary_key]).to_numpy(dtype=np.float32)
    n_parents = parents_X.shape[0]
    parent_index_by_key = pd.Series(np.arange(n_parents), index=parent_keys)

    child_keys = df_children_encoded[children_foreign_key].to_numpy()
    children_X = df_children_encoded.drop(columns=[children_foreign_key]).to_numpy(dtype=np.float32)
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
    labels = np.concatenate([pos_labels, neg_labels]).astype(np.float32, copy=False)

    t1 = time.time()
    print(f"prepare_training_data() | time: {t1 - t0:.2f}s")
    return parent_vecs, child_vecs, labels


def train(
    *,
    model: ParentChildMatcher,
    parent_vecs: np.ndarray,
    child_vecs: np.ndarray,
    labels: np.ndarray,
    do_plot_losses: bool = True,
) -> None:
    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    max_epochs = 1000

    X_parent = torch.tensor(parent_vecs, dtype=torch.float32)
    X_child = torch.tensor(child_vecs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

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
            optimizer.zero_grad()
            pred = model(batch_parent, batch_child)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_child.size(0)
        train_loss /= train_size
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_parent, batch_child, batch_y in val_loader:
                pred = model(batch_parent, batch_child)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_child.size(0)
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


def store_model(*, model: ParentChildMatcher, smart_select_workspace_dir: Path) -> None:
    smart_select_workspace_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "parent_dim": model.parent_dim,
        "child_dim": model.child_dim,
        "hidden_dim": model.hidden_dim,
        "emb_dim": model.emb_dim,
    }
    model_config_path = smart_select_workspace_dir / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=4))
    model_state_path = smart_select_workspace_dir / "model_weights.pt"
    torch.save(model.state_dict(), model_state_path)


def load_model(*, smart_select_workspace_dir: Path) -> ParentChildMatcher:
    model_config_path = smart_select_workspace_dir / "model_config.json"
    model_config = json.loads(model_config_path.read_text())
    model = ParentChildMatcher(
        parent_dim=model_config["parent_dim"],
        child_dim=model_config["child_dim"],
        hidden_dim=model_config["hidden_dim"],
        emb_dim=model_config["emb_dim"],
    )
    model_state_path = smart_select_workspace_dir / "model_weights.pt"
    model.load_state_dict(torch.load(model_state_path))
    return model


def infer_best_parent(
    *,
    model: ParentChildMatcher,
    tgt_encoded: pd.DataFrame,
    parent_encoded: pd.DataFrame,
) -> torch.Tensor:
    t0 = time.time()
    tgt_vecs = torch.tensor(tgt_encoded.values.astype(np.float32))
    parent_vecs = torch.tensor(parent_encoded.values.astype(np.float32))
    n_tgt = tgt_vecs.shape[0]
    n_parent = parent_vecs.shape[0]
    tgt_vecs_expanded = tgt_vecs.repeat_interleave(n_parent, dim=0)
    parent_vecs_expanded = parent_vecs.repeat(n_tgt, 1)
    model.eval()
    with torch.no_grad():
        scores = model(parent_vecs_expanded, tgt_vecs_expanded).squeeze()
        score_matrix = scores.view(n_tgt, n_parent)
        best_parent_indices = torch.argmax(score_matrix, dim=1).cpu().numpy()
        t1 = time.time()
    print(f"infer_best_parent() | time: {t1 - t0:.2f}s")
    return best_parent_indices
