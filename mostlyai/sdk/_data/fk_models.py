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
        sub_column_embedding_dim: int = 32,
        entity_hidden_dim: int = 256,
        entity_embedding_dim: int = 16,
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
        sub_column_embedding_dim: int = 32,
        entity_hidden_dim: int = 256,
        entity_embedding_dim: int = 16,
        similarity_hidden_dim: int = 256,
        num_classes: int = 3,
    ):
        super().__init__()
        self.entity_embedding_dim = entity_embedding_dim
        self.similarity_hidden_dim = similarity_hidden_dim
        self.num_classes = num_classes  # 0=match, 1=negative, 2=null
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
        # output 3 classes: [match_prob, negative_prob, null_prob]
        self.classifier = nn.Sequential(
            nn.Linear(3 * self.entity_embedding_dim, self.similarity_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.similarity_hidden_dim, self.num_classes),
        )

    def forward(self, parent_inputs: dict[str, torch.Tensor], child_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        parent_encoded = self.parent_encoder(parent_inputs)
        child_encoded = self.child_encoder(child_inputs)

        similarity = F.normalize(parent_encoded, dim=1) * F.normalize(child_encoded, dim=1)  # cosine similarity
        classifier_input = torch.cat([parent_encoded, child_encoded, similarity], dim=1)
        logits = self.classifier(classifier_input)

        return logits  # shape: (batch_size, 3)


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
    n_negative: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepare training data for a parent-child matching model with null FK support.
    For each child, samples will include:
    - One positive pair (correct parent) with label=0 [non-null children only]
    - Multiple negative pairs (wrong parents) with label=1 [non-null children only]
    - Multiple null pairs (any parent) with label=2 [null children only]

    The training set composition respects the actual null ratio in the data.

    Args:
        parent_encoded_data: Encoded parent data
        tgt_encoded_data: Encoded child data
        parent_primary_key: Primary key of parents
        tgt_parent_key: Foreign key of children
        sample_size: Number of children to sample (None = use all)
        n_negative: Number of negative/null samples per child
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

    # separate null and non-null children based on their FK values
    null_mask = pd.isna(child_keys)
    null_children_X = children_X[null_mask]
    non_null_children_X = children_X[~null_mask]
    non_null_child_keys = child_keys[~null_mask]

    n_null = len(null_children_X)
    n_non_null = len(non_null_children_X)

    # map each non-null child to its true parent row index
    if n_non_null > 0:
        true_parent_pos = parent_index_by_key.loc[non_null_child_keys].to_numpy()
        if np.any(pd.isna(true_parent_pos)):
            raise ValueError("Some child foreign keys do not match any parent primary key")
    else:
        true_parent_pos = np.array([], dtype=np.int64)

    # Lists to accumulate data
    parent_vecs_list = []
    child_vecs_list = []
    labels_list = []

    # 1. Positive pairs (label=0) - one per non-null child
    if n_non_null > 0:
        pos_parents = parents_X[true_parent_pos]
        parent_vecs_list.append(pos_parents)
        child_vecs_list.append(non_null_children_X)
        labels_list.append(np.zeros(n_non_null, dtype=np.int64))

    # 2. Negative pairs (label=1) - n_negative per non-null child (vectorized)
    if n_non_null > 0:
        # Generate all negative samples at once
        neg_indices = rng.integers(0, n_parents, size=(n_non_null, n_negative))

        # Ensure negatives are not the true parent
        true_parent_pos_expanded = true_parent_pos[:, np.newaxis]  # Shape: (n_non_null, 1)
        mask = neg_indices == true_parent_pos_expanded

        # Replace any matches with new random samples
        while mask.any():
            neg_indices[mask] = rng.integers(0, n_parents, size=mask.sum())
            mask = neg_indices == true_parent_pos_expanded

        # Flatten and create pairs
        neg_parents = parents_X[neg_indices.ravel()]  # Shape: (n_non_null * n_negative, n_features)
        neg_children = np.repeat(non_null_children_X, n_negative, axis=0)  # Same shape

        parent_vecs_list.append(neg_parents)
        child_vecs_list.append(neg_children)
        labels_list.append(np.ones(n_non_null * n_negative, dtype=np.int64))

    # 3. Null pairs (label=2) - n_negative per null child
    # Pair each null child with random parents to teach: "this child doesn't belong to any parent"
    if n_null > 0:
        # Generate random parent indices for all null children at once
        null_parent_indices = rng.integers(0, n_parents, size=(n_null, n_negative))

        # Flatten and create pairs
        null_parents = parents_X[null_parent_indices.ravel()]  # Shape: (n_null * n_negative, n_features)
        null_children_repeated = np.repeat(null_children_X, n_negative, axis=0)  # Same shape

        parent_vecs_list.append(null_parents)
        child_vecs_list.append(null_children_repeated)
        labels_list.append(np.full(n_null * n_negative, 2, dtype=np.int64))

    # concatenate all data
    parent_vecs = np.vstack(parent_vecs_list).astype(np.float32, copy=False)
    child_vecs = np.vstack(child_vecs_list).astype(np.float32, copy=False)
    labels_vec = np.concatenate(labels_list).astype(np.int64, copy=False)

    # convert to pandas
    parent_pd = pd.DataFrame(parent_vecs, columns=parent_encoded_data.drop(columns=[parent_primary_key]).columns)
    child_pd = pd.DataFrame(child_vecs, columns=tgt_encoded_data.drop(columns=[tgt_parent_key]).columns)
    labels_pd = pd.Series(labels_vec, name="labels")

    t1 = time.time()
    print(f"prepare_training_data() | time: {t1 - t0:.2f}s")
    print(f"  - Non-null children: {n_non_null} ({n_non_null / sample_size * 100:.1f}%)")
    print(f"  - Null children: {n_null} ({n_null / sample_size * 100:.1f}%)")
    print(f"  - Positive pairs (label=0): {(labels_vec == 0).sum()}")
    print(f"  - Negative pairs (label=1): {(labels_vec == 1).sum()}")
    print(f"  - Null pairs (label=2): {(labels_vec == 2).sum()}")
    return parent_pd, child_pd, labels_pd


def train(
    *,
    model: ParentChildMatcher,
    parent_pd: pd.DataFrame,
    child_pd: pd.DataFrame,
    labels: pd.Series,
    do_plot_losses: bool = True,
) -> None:
    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    max_epochs = 1000

    X_parent = torch.tensor(parent_pd.values, dtype=torch.int64)
    X_child = torch.tensor(child_pd.values, dtype=torch.int64)
    y = torch.tensor(labels.values, dtype=torch.int64)  # Labels are now class indices (0, 1, 2)
    dataset = TensorDataset(X_parent, X_child, y)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropy for multi-class classification

    train_losses, val_losses = [], []
    best_model_state = None

    for epoch in range(max_epochs):
        # training phase
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_parent, batch_child, batch_y in train_loader:
            batch_parent = {col: batch_parent[:, i] for i, col in enumerate(parent_pd.columns)}
            batch_child = {col: batch_child[:, i] for i, col in enumerate(child_pd.columns)}
            optimizer.zero_grad()
            logits = model(batch_parent, batch_child)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_y.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        train_loss /= train_size
        train_acc = train_correct / train_size
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch_parent, batch_child, batch_y in val_loader:
                batch_parent = {col: batch_parent[:, i] for i, col in enumerate(parent_pd.columns)}
                batch_child = {col: batch_child[:, i] for i, col in enumerate(child_pd.columns)}
                logits = model(batch_parent, batch_child)
                loss = loss_fn(logits, batch_y)
                val_loss += loss.item() * batch_y.size(0)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        val_loss /= val_size
        val_acc = val_correct / val_size
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{max_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build probability matrices for parent-child matching.

    Returns:
        match_prob_matrix: (n_tgt, n_parent) - probability each parent is a match
        null_prob: (n_tgt,) - probability each child should have null FK
    """
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
        logits = model(parent_inputs, tgt_inputs)  # Shape: (n_tgt * n_parent, 3)
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities

        # Reshape to (n_tgt, n_parent, 3)
        probs = probs.view(n_tgt, n_parent, 3)

        # Extract match probabilities (class 0) for each (child, parent) pair
        match_prob_matrix = probs[:, :, 0]  # Shape: (n_tgt, n_parent)

        # Extract null probabilities (class 2) - average across all parents for each child
        # The null probability should be consistent regardless of which parent we pair with
        # Taking mean gives us the model's overall belief that this child should have null FK
        null_prob = probs[:, :, 2].mean(dim=1)  # Shape: (n_tgt,)

        t1 = time.time()
    print(f"build_parent_child_probabilities() | time: {t1 - t0:.2f}s")
    return match_prob_matrix, null_prob


def sample_best_parents(
    *,
    match_prob_matrix: torch.Tensor,
    null_prob: torch.Tensor,
    sample_probabilistically: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> np.ndarray:
    """
    Sample best parent for each child, or None if should be null.

    Args:
        match_prob_matrix: (n_tgt, n_parent) probability each parent is a match
        null_prob: (n_tgt,) probability each child should have null FK
        sample_probabilistically: If True, sample based on null_prob to preserve distribution.
                                  If False, use argmax (deterministic).
        temperature: Controls variance in parent selection (default=1.0)
                    - temperature=0.0: Always pick argmax (most confident match)
                    - temperature=1.0: Sample from original probabilities
                    - temperature>1.0: Increase variance (flatten distribution)
                    Higher values create more diverse matches but may reduce quality.
        top_k: If specified, only sample from top-K most probable parents per child.
               This prevents unrealistic outlier matches while maintaining variance.
               Recommended: 10-50 depending on parent pool size.

    Returns:
        best_parent_indices: Array of parent indices, or -1 for null FK
    """
    n_tgt = match_prob_matrix.shape[0]
    best_parent_indices = np.full(n_tgt, -1, dtype=np.int64)

    rng = np.random.default_rng()

    # For each child, decide: null or match to a parent?
    for i in range(n_tgt):
        if sample_probabilistically:
            # Probabilistic sampling: preserves the learned null distribution
            # Sample from bernoulli distribution with p=null_prob[i]
            is_null = rng.random() < null_prob[i].cpu().numpy()
            if is_null:
                best_parent_indices[i] = -1
            else:
                # Sample parent based on match probabilities with temperature
                if temperature == 0.0:
                    # Deterministic: pick best match
                    best_parent_indices[i] = torch.argmax(match_prob_matrix[i]).cpu().numpy()
                else:
                    # Apply top-k filtering if specified
                    probs = match_prob_matrix[i]
                    candidate_indices = torch.arange(len(probs))

                    if top_k is not None and top_k < len(probs):
                        # Keep only top-k most probable parents
                        top_k_values, top_k_indices = torch.topk(probs, k=top_k)
                        probs = top_k_values
                        candidate_indices = top_k_indices

                    # Probabilistic: sample from distribution with temperature scaling
                    # Higher temperature = more uniform sampling (more variance)
                    logits = torch.log(probs + 1e-10) / temperature
                    probs = torch.softmax(logits, dim=0).cpu().numpy()

                    # Sample from candidates and map back to original indices
                    sampled_candidate = rng.choice(len(probs), p=probs)
                    best_parent_indices[i] = candidate_indices[sampled_candidate].cpu().numpy()
        else:
            # Deterministic: use threshold of 0.5
            if null_prob[i] > 0.5:
                best_parent_indices[i] = -1
            else:
                best_parent_indices[i] = torch.argmax(match_prob_matrix[i]).cpu().numpy()

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
