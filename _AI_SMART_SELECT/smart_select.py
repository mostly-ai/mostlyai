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
from typing import Literal

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
    foreign_key: str | None = None,
    pre_training_dir: Path,
) -> None:
    t0 = time.time()

    pre_training_dir.mkdir(parents=True, exist_ok=True)

    key_columns = [col for col in (primary_key, foreign_key) if col is not None]
    data_columns = df.columns.difference(key_columns)
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
        "foreign_key": foreign_key,
    }
    pre_training_meta_path = pre_training_dir / "pre_training_meta.json"
    pre_training_meta_path.write_text(json.dumps(pre_training_meta, indent=4))

    t1 = time.time()
    print(f"pre_training() | time: {t1 - t0:.2f}s")


def encode_df(
    *, df: pd.DataFrame, pre_training_dir: Path, drop_primary_key: bool = False, drop_foreign_key: bool = False
) -> pd.DataFrame:
    t0 = time.time()

    pre_training_meta_path = pre_training_dir / "pre_training_meta.json"
    pre_training_meta = json.loads(pre_training_meta_path.read_text())
    primary_key = pre_training_meta["primary_key"]
    foreign_key = pre_training_meta["foreign_key"]

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
    for key, drop_key in [(primary_key, drop_primary_key), (foreign_key, drop_foreign_key)]:
        if key is not None and not drop_key:
            data.insert(0, df[[key]])
            features.insert(0, key)
    data = np.concatenate(data, axis=1)

    t1 = time.time()
    print(f"encode_df() | time: {t1 - t0:.2f}s")
    return pd.DataFrame(data, columns=features)


def prepare_training_data(
    df_parent_encoded: pd.DataFrame,
    df_child_encoded: pd.DataFrame,
    parent_primary_key: str,
    child_foreign_key: str,
    n_children: int | None = None,
    n_false_parents: int = 1,
    negative_sampling_strategy: Literal["random", "hard"] = "random",
    model: ParentChildMatcher | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0 = time.time()
    if n_children is None:
        n_children = len(df_child_encoded)

    # Build parent feature matrix and key->row mapping
    parent_keys = df_parent_encoded[parent_primary_key].to_numpy()
    parent_X = df_parent_encoded.drop(columns=[parent_primary_key]).to_numpy(dtype=np.float32)
    num_parents = parent_X.shape[0]
    parent_index_by_key = pd.Series(np.arange(num_parents), index=parent_keys)

    # Build child feature matrix and foreign keys
    child_keys = df_child_encoded[child_foreign_key].to_numpy()
    child_X_full = df_child_encoded.drop(columns=[child_foreign_key]).to_numpy(dtype=np.float32)
    num_children_total = child_X_full.shape[0]

    # Sample children without replacement
    n_children = min(int(n_children), num_children_total)
    rng = np.random.default_rng()
    sampled_child_indices = rng.choice(num_children_total, size=n_children, replace=False)
    child_X = child_X_full[sampled_child_indices]
    sampled_child_keys = child_keys[sampled_child_indices]

    # Map each sampled child to its true parent row index
    true_parent_pos = parent_index_by_key.loc[sampled_child_keys].to_numpy()
    if np.any(pd.isna(true_parent_pos)):
        raise ValueError("Some child foreign keys do not match any parent primary key")
    true_parent_pos = true_parent_pos.astype(np.int64, copy=False)

    # Positive pairs
    pos_parent = parent_X[true_parent_pos]
    pos_child = child_X
    pos_labels = np.ones(n_children, dtype=np.float32)

    # Prepare negatives
    if negative_sampling_strategy == "random":
        # Vectorized random negative sampling excluding true parent index
        if n_false_parents <= 0:
            neg_parent = np.empty((0, parent_X.shape[1]), dtype=np.float32)
            neg_child = np.empty((0, child_X.shape[1]), dtype=np.float32)
            neg_labels = np.empty((0,), dtype=np.float32)
        else:
            neg_indices = rng.integers(0, num_parents, size=(n_children, n_false_parents))
            # Resample any collisions with true parent indices
            mask = neg_indices == true_parent_pos[:, None]
            while mask.any():
                neg_indices[mask] = rng.integers(0, num_parents, size=mask.sum())
                mask = neg_indices == true_parent_pos[:, None]
            neg_parent = parent_X[neg_indices.reshape(-1)]
            neg_child = np.repeat(child_X, repeats=n_false_parents, axis=0)
            neg_labels = np.zeros(n_children * n_false_parents, dtype=np.float32)
    else:
        # Batched hard negatives: pick the k lowest-similarity parents per child (excluding the true parent)
        assert model is not None, "Model must be provided for hard negative sampling"
        model.eval()
        hard_k = max(1, int(n_false_parents))

        # Torch tensors
        parent_tensor = torch.tensor(parent_X, dtype=torch.float32)
        # Chunk over parents to keep memory bounded
        parent_chunk_size = 4096

        neg_indices_list: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(n_children):
                child_vec = torch.tensor(child_X[i], dtype=torch.float32).unsqueeze(0)
                # Accumulate similarities for all parents
                sims = np.empty(num_parents, dtype=np.float32)
                start = 0
                while start < num_parents:
                    end = min(start + parent_chunk_size, num_parents)
                    parent_batch = parent_tensor[start:end]
                    child_batch = child_vec.expand(end - start, -1)
                    scores = model(parent_batch, child_batch).squeeze(1).cpu().numpy().astype(np.float32, copy=False)
                    sims[start:end] = scores
                    start = end
                # Exclude true parent
                sims[true_parent_pos[i]] = np.inf

                # Prefer those below threshold if enough exist; else take global minima
                threshold = 0.1
                below = np.flatnonzero(sims < threshold)
                if below.size >= hard_k:
                    # Choose the smallest scores among those below threshold
                    if below.size == hard_k:
                        chosen = below
                    else:
                        part = np.argpartition(sims[below], kth=hard_k - 1)[:hard_k]
                        chosen = below[part]
                else:
                    # Take the k global minima
                    if num_parents - 1 == hard_k:
                        # All except the true parent
                        chosen = np.setdiff1d(
                            np.arange(num_parents), np.array([true_parent_pos[i]]), assume_unique=True
                        )
                    else:
                        chosen = np.argpartition(sims, kth=hard_k - 1)[:hard_k]
                neg_indices_list.append(chosen.astype(np.int64, copy=False))

        neg_indices = np.stack(neg_indices_list, axis=0)
        neg_parent = parent_X[neg_indices.reshape(-1)]
        neg_child = np.repeat(child_X, repeats=hard_k, axis=0)
        neg_labels = np.zeros(n_children * hard_k, dtype=np.float32)

    # Concatenate positives and negatives
    parent_vecs = np.vstack([pos_parent, neg_parent]).astype(np.float32, copy=False)
    child_vecs = np.vstack([pos_child, neg_child]).astype(np.float32, copy=False)
    labels = np.concatenate([pos_labels, neg_labels]).astype(np.float32, copy=False)

    t1 = time.time()
    print(f"prepare_training_data() | time: {t1 - t0:.2f}s")
    return parent_vecs, child_vecs, labels


def train(
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


def store_model(model: ParentChildMatcher, smart_select_workspace_dir: Path) -> None:
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


def load_model(smart_select_workspace_dir: Path) -> ParentChildMatcher:
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


def infer_best_non_ctx(
    *,
    model: ParentChildMatcher,
    tgt_encoded: pd.DataFrame,
    non_ctx_encoded: pd.DataFrame,
) -> torch.Tensor:
    t0 = time.time()
    tgt_vecs = torch.tensor(tgt_encoded.values.astype(np.float32))
    non_ctx_vecs = torch.tensor(non_ctx_encoded.values.astype(np.float32))
    n_tgt = tgt_vecs.shape[0]
    n_non_ctx = non_ctx_vecs.shape[0]
    tgt_vecs_expanded = tgt_vecs.repeat_interleave(n_non_ctx, dim=0)
    non_ctx_vecs_expanded = non_ctx_vecs.repeat(n_tgt, 1)
    model.eval()
    with torch.no_grad():
        scores = model(non_ctx_vecs_expanded, tgt_vecs_expanded).squeeze()
        score_matrix = scores.view(n_tgt, n_non_ctx)
        best_non_ctx_indices = torch.argmax(score_matrix, dim=1).cpu().numpy()
        t1 = time.time()
    print(f"infer_best_non_ctx() | time: {t1 - t0:.2f}s")
    return best_non_ctx_indices
