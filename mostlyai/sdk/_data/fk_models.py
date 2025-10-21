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
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathvalidate import sanitize_filename
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from mostlyai.engine._encoding_types.tabular.categorical import (
    analyze_categorical,
    analyze_reduce_categorical,
    encode_categorical,
)
from mostlyai.engine._encoding_types.tabular.datetime import analyze_datetime, analyze_reduce_datetime, encode_datetime
from mostlyai.engine._encoding_types.tabular.numeric import analyze_numeric, analyze_reduce_numeric, encode_numeric
from mostlyai.sdk._data.base import DataTable
from mostlyai.sdk._data.util.common import IS_NULL, NON_CONTEXT_COLUMN_INFIX

_LOG = logging.getLogger(__name__)


# =============================================================================
# GLOBAL HYPERPARAMETER DEFAULTS
# =============================================================================

# Model Architecture Parameters
SUB_COLUMN_EMBEDDING_DIM = 32
ENTITY_HIDDEN_DIM = 256
ENTITY_EMBEDDING_DIM = 16
SIMILARITY_HIDDEN_DIM = 256
PEAKEDNESS_SCALER = 7.0

# Training Parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
MAX_EPOCHS = 1000
PATIENCE = 20
N_NEGATIVE_SAMPLES = 10
VAL_SPLIT = 0.2
DROPOUT_RATE = 0.2
EARLY_STOPPING_DELTA = 1e-5
NUMERICAL_STABILITY_EPSILON = 1e-10

# Data Sampling Parameters
MAX_PARENT_SAMPLE_SIZE = 10000
MAX_TGT_PER_PARENT = 2

# Inference Parameters
TEMPERATURE = 1.0
TOP_K = 200


def safe_name(text: str) -> str:
    safe = sanitize_filename(text)
    digest = hashlib.md5(safe.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


class EntityEncoder(nn.Module):
    def __init__(
        self,
        cardinalities: dict[str, int],
        sub_column_embedding_dim: int = SUB_COLUMN_EMBEDDING_DIM,
        entity_hidden_dim: int = ENTITY_HIDDEN_DIM,
        entity_embedding_dim: int = ENTITY_EMBEDDING_DIM,
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
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.entity_hidden_dim, self.entity_embedding_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.entity_embedding_dim, self.entity_embedding_dim),
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
        sub_column_embedding_dim: int = SUB_COLUMN_EMBEDDING_DIM,
        entity_hidden_dim: int = ENTITY_HIDDEN_DIM,
        entity_embedding_dim: int = ENTITY_EMBEDDING_DIM,
        similarity_hidden_dim: int = SIMILARITY_HIDDEN_DIM,
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

    def forward(self, parent_inputs: dict[str, torch.Tensor], child_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        parent_encoded = self.parent_encoder(parent_inputs)
        child_encoded = self.child_encoder(child_inputs)

        # Compute cosine similarity between parent and child embeddings
        similarity = F.cosine_similarity(parent_encoded, child_encoded, dim=1)

        # Convert to probability (sigmoid to ensure [0,1] range)
        probability = torch.sigmoid(similarity * PEAKEDNESS_SCALER).unsqueeze(1)

        return probability


def get_cardinalities(*, stats_dir: Path) -> dict[str, int]:
    stats_path = stats_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    cardinalities = {
        f"{column}_{sub_column}": cardinality
        for column, column_stats in stats["columns"].items()
        for sub_column, cardinality in column_stats["cardinalities"].items()
    }
    return cardinalities


def analyze_df(
    *,
    df: pd.DataFrame,
    primary_key: str | None = None,
    parent_key: str | None = None,
    data_columns: list[str] | None = None,
    stats_dir: Path,
) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)

    key_columns = []
    if primary_key is not None:
        key_columns.append(primary_key)
    if parent_key is not None:
        key_columns.append(parent_key)

    data_columns = data_columns or list(df.columns)

    # preserve column order to ensure deterministic encoding
    data_columns = [col for col in data_columns if col not in key_columns and col in df.columns]
    num_columns = [col for col in data_columns if col in df.select_dtypes(include="number").columns]
    dt_columns = [col for col in data_columns if col in df.select_dtypes(include="datetime").columns]
    cat_columns = [col for col in data_columns if col not in num_columns + dt_columns]

    # store stats metadata as JSON
    stats = {
        "primary_key": primary_key,
        "parent_key": parent_key,
        "data_columns": data_columns,
        "cat_columns": cat_columns,
        "num_columns": num_columns,
        "dt_columns": dt_columns,
        "columns": {},
    }
    for col in data_columns:
        values = df[col]
        root_keys = pd.Series(np.arange(len(values)), name="root_keys")
        if col in cat_columns:
            analyze, reduce = analyze_categorical, analyze_reduce_categorical
        elif col in num_columns:
            analyze, reduce = analyze_numeric, analyze_reduce_numeric
        elif col in dt_columns:
            analyze, reduce = analyze_datetime, analyze_reduce_datetime
        else:
            raise ValueError(f"unknown column type: {col}")
        col_stats = analyze(values, root_keys)
        col_stats = reduce([col_stats], value_protection=True)
        stats["columns"][col] = col_stats

    # store stats
    stats_path = stats_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=4))


def encode_df(
    *, df: pd.DataFrame, stats_dir: Path, include_primary_key: bool = True, include_parent_key: bool = True
) -> pd.DataFrame:
    # load stats
    stats_path = stats_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    primary_key = stats["primary_key"]
    parent_key = stats["parent_key"]
    cat_columns = stats["cat_columns"]
    num_columns = stats["num_columns"]
    dt_columns = stats["dt_columns"]

    # encode columns
    data = []
    for col, col_stats in stats["columns"].items():
        if col in cat_columns:
            encode = encode_categorical
        elif col in num_columns:
            encode = encode_numeric
        elif col in dt_columns:
            encode = encode_datetime
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

    return data


# FK Training Data Pull Functions


def fetch_parent_data(parent_table: DataTable, max_sample_size: int = MAX_PARENT_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Fetch unique parent data with optional sampling limit.

    Reads the parent table in chunks to efficiently collect unique parent records
    until the maximum sample size is reached. Stops early once the limit is met
    to avoid unnecessary data processing.

    Args:
        parent_table: Parent table to extract data from. Must have a primary key defined.
        max_sample_size: Maximum number of unique records to collect. Defaults to 10,000.

    Returns:
        DataFrame containing complete parent records with all columns.
        Records are unique by primary key.
    """
    primary_key = parent_table.primary_key
    seen_keys = set()
    collected_rows = []

    for chunk_df in parent_table.read_chunks(columns=parent_table.columns, do_coerce_dtypes=True):
        # Drop duplicates in this chunk
        chunk_df = chunk_df.drop_duplicates(subset=[primary_key])

        # Add rows to list until we have enough
        for _, row in chunk_df.iterrows():
            key = row[primary_key]
            if key not in seen_keys:
                seen_keys.add(key)
                collected_rows.append(row)
                if len(collected_rows) >= max_sample_size:
                    break

        if len(collected_rows) >= max_sample_size:
            break

    parent_data = pd.DataFrame(collected_rows).reset_index(drop=True)
    return parent_data


def fetch_tgt_data(
    *,
    tgt_table: DataTable,
    tgt_parent_key: str,
    parent_keys: list,
    max_tgt_per_parent: int = MAX_TGT_PER_PARENT,
) -> pd.DataFrame:
    """
    Fetch target data with per-parent limits.

    Reads target table in chunks and tracks how many target records each parent has.
    Stops adding target records for a parent once the limit is reached.

    Args:
        tgt_table: Target table to fetch from.
        tgt_parent_key: Foreign key column in target table.
        parent_keys: List of parent key values to filter by.
        max_tgt_per_parent: Maximum target records per parent. Defaults to 1.

    Returns:
        DataFrame containing target records, limited by max_tgt_per_parent constraint.
    """

    # Track count of children per parent and collect rows directly
    parent_counts = defaultdict(int)
    collected_rows = []
    where = {tgt_parent_key: parent_keys}

    for chunk_df in tgt_table.read_chunks(where=where, columns=tgt_table.columns, do_coerce_dtypes=True):
        if len(chunk_df) == 0:
            continue

        for _, row in chunk_df.iterrows():
            parent_id = row[tgt_parent_key]

            # Add child only if under the limit
            if parent_counts[parent_id] < max_tgt_per_parent:
                collected_rows.append(row)
                parent_counts[parent_id] += 1

    # Convert to DataFrame
    child_data = pd.DataFrame(collected_rows).reset_index(drop=True)
    _LOG.info(f"fetch_child_data | fetched: {len(child_data)}")
    return child_data


def pull_fk_model_training_data(
    *,
    tgt_table: DataTable,
    tgt_parent_key: str,
    parent_table: DataTable,
    parent_primary_key: str,
    max_parent_sample_size: int = MAX_PARENT_SAMPLE_SIZE,
    max_tgt_per_parent: int = MAX_TGT_PER_PARENT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull training data for a specific non-context FK relation.

    Args:
        schema: Schema containing the tables
        tgt_parent_key: Foreign key column in target table
        parent_table: Parent table
        parent_primary_key: Primary key column in parent table
        max_parent_sample_size: Maximum parent keys to sample
        max_tgt_per_parent: Maximum target records per parent

    Returns:
        Tuple of (parent_data, tgt_data)
    """
    parent_data = fetch_parent_data(parent_table=parent_table, max_sample_size=max_parent_sample_size)
    parent_keys_list = parent_data[parent_primary_key].tolist() if not parent_data.empty else []
    tgt_data = fetch_tgt_data(
        tgt_table=tgt_table,
        tgt_parent_key=tgt_parent_key,
        parent_keys=parent_keys_list,
        max_tgt_per_parent=max_tgt_per_parent,
    )
    return parent_data, tgt_data


def prepare_training_data(
    parent_encoded_data: pd.DataFrame,
    tgt_encoded_data: pd.DataFrame,
    parent_primary_key: str,
    tgt_parent_key: str,
    sample_size: int | None = None,
    n_negative: int = N_NEGATIVE_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepare training data for a parent-child matching model.
    For each non-null child, samples will include:
    - One positive pair (correct parent) with label=1
    - Multiple negative pairs (wrong parents) with label=0

    Null children are excluded from training - nulls will be handled via _is_null column during inference.

    Args:
        parent_encoded_data: Encoded parent data
        tgt_encoded_data: Encoded child data
        parent_primary_key: Primary key of parents
        tgt_parent_key: Foreign key of children
        sample_size: Number of children to sample (None = use all)
        n_negative: Number of negative samples per child
    """
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

    # filter out null children - they will be handled by _is_null column during inference
    non_null_mask = ~pd.isna(child_keys)
    children_X = children_X[non_null_mask]
    child_keys = child_keys[non_null_mask]
    n_non_null = len(children_X)

    if n_non_null == 0:
        raise ValueError("No non-null children found in training data")

    # map each non-null child to its true parent row index
    true_parent_pos = parent_index_by_key.loc[child_keys].to_numpy()
    if np.any(pd.isna(true_parent_pos)):
        raise ValueError("Some child foreign keys do not match any parent primary key")

    # 1. Positive pairs (label=1) - one per non-null child
    pos_parents = parents_X[true_parent_pos]
    pos_labels = np.ones(n_non_null, dtype=np.float32)

    # 2. Negative pairs (label=0) - n_negative per non-null child (vectorized)
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
    neg_children = np.repeat(children_X, n_negative, axis=0)  # Same shape
    neg_labels = np.zeros(n_non_null * n_negative, dtype=np.float32)

    # concatenate positives and negatives
    parent_vecs = np.vstack([pos_parents, neg_parents]).astype(np.float32, copy=False)
    child_vecs = np.vstack([children_X, neg_children]).astype(np.float32, copy=False)
    labels_vec = np.concatenate([pos_labels, neg_labels]).astype(np.float32, copy=False)

    # convert to pandas
    parent_pd = pd.DataFrame(parent_vecs, columns=parent_encoded_data.drop(columns=[parent_primary_key]).columns)
    child_pd = pd.DataFrame(child_vecs, columns=tgt_encoded_data.drop(columns=[tgt_parent_key]).columns)
    labels_pd = pd.Series(labels_vec, name="labels")

    return parent_pd, child_pd, labels_pd


def train(
    *,
    model: ParentChildMatcher,
    parent_pd: pd.DataFrame,
    child_pd: pd.DataFrame,
    labels: pd.Series,
) -> None:
    patience = PATIENCE
    best_val_loss = float("inf")
    epochs_no_improve = 0
    max_epochs = MAX_EPOCHS

    X_parent = torch.tensor(parent_pd.values, dtype=torch.int64)
    X_child = torch.tensor(child_pd.values, dtype=torch.int64)
    y = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_parent, X_child, y)

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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

        _LOG.info(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # early stopping check
        if val_loss < best_val_loss - EARLY_STOPPING_DELTA:  # small delta to avoid float issues
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                _LOG.info(f"Early stopping at epoch {epoch + 1}")
                break

    assert best_model_state is not None
    model.load_state_dict(best_model_state)
    _LOG.info("Best model restored (lowest validation loss).")


def store_fk_model(*, model: ParentChildMatcher, fk_model_workspace_dir: Path) -> None:
    fk_model_workspace_dir.mkdir(parents=True, exist_ok=True)
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
    model_config_path = fk_model_workspace_dir / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=4))
    model_state_path = fk_model_workspace_dir / "model_weights.pt"
    torch.save(model.state_dict(), model_state_path)


def load_fk_model(*, fk_model_workspace_dir: Path) -> ParentChildMatcher:
    model_config_path = fk_model_workspace_dir / "model_config.json"
    model_config = json.loads(model_config_path.read_text())
    model = ParentChildMatcher(
        parent_cardinalities=model_config["parent_encoder"]["cardinalities"],
        child_cardinalities=model_config["child_encoder"]["cardinalities"],
        sub_column_embedding_dim=model_config["parent_encoder"]["sub_column_embedding_dim"],
        entity_hidden_dim=model_config["parent_encoder"]["entity_hidden_dim"],
        entity_embedding_dim=model_config["parent_encoder"]["entity_embedding_dim"],
        similarity_hidden_dim=model_config["similarity_hidden_dim"],
    )
    model_state_path = fk_model_workspace_dir / "model_weights.pt"
    model.load_state_dict(torch.load(model_state_path))
    return model


def build_parent_child_probabilities(
    *,
    model: ParentChildMatcher,
    tgt_encoded: pd.DataFrame,
    parent_encoded: pd.DataFrame,
) -> torch.Tensor:
    """
    Build probability matrix for parent-child matching.

    Args:
        model: Trained parent-child matching model
        tgt_encoded: Encoded target/child data (C rows)
        parent_encoded: Encoded parent data (Cp rows - assigned parent batch)

    Returns:
        prob_matrix: (C, Cp) - probability each parent candidate is a match for each child
    """
    n_tgt = tgt_encoded.shape[0]
    n_parent_batch = parent_encoded.shape[0]

    tgt_inputs = {col: torch.tensor(tgt_encoded[col].values.astype(np.int64)) for col in tgt_encoded.columns}
    parent_inputs = {col: torch.tensor(parent_encoded[col].values.astype(np.int64)) for col in parent_encoded.columns}

    model.eval()
    with torch.no_grad():
        # Compute embeddings once for each unique child and parent
        child_embeddings = model.child_encoder(tgt_inputs)  # Shape: (C, embedding_dim)
        parent_embeddings = model.parent_encoder(parent_inputs)  # Shape: (Cp, embedding_dim)

        # Each child with all parent candidates: C repeated Cp times, Cp repeated C times
        child_embeddings_interleaved = child_embeddings.repeat_interleave(
            n_parent_batch, dim=0
        )  # Shape: (C*Cp, embedding_dim)
        parent_embeddings_interleaved = parent_embeddings.repeat(n_tgt, 1)  # Shape: (Cp*C, embedding_dim)

        # Compute probabilities using cosine similarity
        similarity = F.cosine_similarity(parent_embeddings_interleaved, child_embeddings_interleaved, dim=1)
        probs = torch.sigmoid(similarity * PEAKEDNESS_SCALER)  # Same scaling as in forward pass

        prob_matrix = probs.view(n_tgt, n_parent_batch)
        return prob_matrix


def sample_best_parents(
    *,
    prob_matrix: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 100,
) -> np.ndarray:
    """
    Sample best parent for each child based on match probabilities.

    Args:
        prob_matrix: (n_tgt, n_parent) probability each parent is a match
        temperature: Controls variance in parent selection (default=1.0)
                    - temperature=0.0: Always pick argmax (most confident match)
                    - temperature=1.0: Sample from original probabilities
                    - temperature>1.0: Increase variance (flatten distribution)
                    Higher values create more diverse matches but may reduce quality.
        top_k: If specified, only sample from top-K most probable parents per child.
               This prevents unrealistic outlier matches while maintaining variance.
               Recommended: 10-50 depending on parent pool size.

    Returns:
        best_parent_indices: Array of parent indices for each child
    """
    n_tgt = prob_matrix.shape[0]
    best_parent_indices = np.full(n_tgt, -1, dtype=np.int64)

    rng = np.random.default_rng()

    for i in range(n_tgt):
        if temperature == 0.0:
            # Deterministic: pick best match
            best_parent_indices[i] = torch.argmax(prob_matrix[i]).cpu().numpy()
        else:
            # Apply top-k filtering if specified
            probs = prob_matrix[i]
            candidate_indices = torch.arange(len(probs))

            if top_k is not None and top_k < len(probs):
                # Keep only top-k most probable parents
                top_k_values, top_k_indices = torch.topk(probs, k=top_k)
                probs = top_k_values
                candidate_indices = top_k_indices

            # Probabilistic: sample from distribution with temperature scaling
            # Higher temperature = more uniform sampling (more variance)
            logits = torch.log(probs + NUMERICAL_STABILITY_EPSILON) / temperature
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            # Sample from candidates and map back to original indices
            sampled_candidate = rng.choice(len(probs), p=probs)
            best_parent_indices[i] = candidate_indices[sampled_candidate].cpu().numpy()

    return best_parent_indices


def match_non_context(
    *,
    fk_models_workspace_dir: Path,
    tgt_data: pd.DataFrame,
    parent_data: pd.DataFrame,
    tgt_parent_key: str,
    parent_primary_key: str,
    parent_table_name: str,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    # Check for _is_null column to determine which rows should have null FK
    # Column name format: {fk_name}.{parent_table_name}._is_null
    is_null_col = NON_CONTEXT_COLUMN_INFIX.join([tgt_parent_key, parent_table_name, IS_NULL])
    has_is_null = is_null_col in tgt_data.columns

    # Initialize FK column with nulls
    tgt_data[tgt_parent_key] = pd.NA

    if has_is_null:
        # Use _is_null column to determine which rows should have null FK
        # _is_null column contains string values "True" or "False"
        is_null_values = tgt_data[is_null_col].astype(str)
        null_mask = is_null_values == "True"
        non_null_mask = ~null_mask

        _LOG.info(
            f"FK matching data | total_rows: {len(tgt_data)} | null_rows: {null_mask.sum()} | non_null_rows: {non_null_mask.sum()}"
        )

        # Only process non-null rows
        if non_null_mask.sum() == 0:
            _LOG.warning(f"All rows have null FK values (via {is_null_col})")
            # Remove _is_null column
            if is_null_col in tgt_data.columns:
                tgt_data = tgt_data.drop(columns=[is_null_col])
            return tgt_data

        # Get indices of non-null rows
        non_null_indices = tgt_data.index[non_null_mask].tolist()

        # Filter to only non-null rows for FK model processing
        tgt_data_non_null = tgt_data.loc[non_null_mask].copy().reset_index(drop=True)

        # Remove _is_null column from data before encoding (it shouldn't be used by the FK model)
        if is_null_col in tgt_data_non_null.columns:
            tgt_data_non_null = tgt_data_non_null.drop(columns=[is_null_col])
    else:
        _LOG.info(f"FK matching data | total_rows: {len(tgt_data)} | null_rows: 0 | non_null_rows: {len(tgt_data)}")
        tgt_data_non_null = tgt_data.copy()
        non_null_indices = tgt_data.index.tolist()
        non_null_mask = pd.Series(True, index=tgt_data.index)

    # Prepare data for FK model
    fk_model_workspace_dir = fk_models_workspace_dir / safe_name(tgt_parent_key)
    tgt_stats_dir = fk_model_workspace_dir / "tgt-stats"
    parent_stats_dir = fk_model_workspace_dir / "parent-stats"

    # Encode target and parent data
    tgt_encoded = encode_df(
        df=tgt_data_non_null,
        stats_dir=tgt_stats_dir,
        include_primary_key=False,
        include_parent_key=False,
    )
    parent_encoded = encode_df(
        df=parent_data,
        stats_dir=parent_stats_dir,
        include_primary_key=False,
    )

    model = load_fk_model(fk_model_workspace_dir=fk_model_workspace_dir)

    # Build probability matrix
    fk_parent_sample_size = len(parent_encoded)
    _LOG.info(
        f"FK model matching | temperature: {temperature} | top_k: {top_k} | parent_sample_size: {fk_parent_sample_size}"
    )

    # Compute parent-child probabilities using new cosine similarity architecture
    prob_matrix = build_parent_child_probabilities(
        model=model,
        tgt_encoded=tgt_encoded,
        parent_encoded=parent_encoded,
    )

    # Sample best parents based on probabilities
    best_parent_indices = sample_best_parents(
        prob_matrix=prob_matrix,
        temperature=temperature,
        top_k=top_k,
    )

    # Map indices to parent IDs
    best_parent_ids = parent_data.iloc[best_parent_indices][parent_primary_key].values

    # Create a Series with the correct index alignment
    parent_ids_series = pd.Series(best_parent_ids, index=non_null_indices)

    # Assign to non-null rows
    tgt_data.loc[non_null_indices, tgt_parent_key] = parent_ids_series

    # Remove _is_null column if it exists
    if has_is_null and is_null_col in tgt_data.columns:
        tgt_data = tgt_data.drop(columns=[is_null_col])

    n_matched = non_null_mask.sum()
    n_null = (~non_null_mask).sum()
    _LOG.info(f"FK matching completed | matched: {n_matched} | null: {n_null}")

    return tgt_data
