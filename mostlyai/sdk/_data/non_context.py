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

"""
Non-context foreign key handling module.

This module provides functionality for handling non-context foreign keys in two modes:
1. Pull-phase: Mark which FKs should be null during data extraction
2. Assignment phase:
   - ML-based: Use trained neural network models for intelligent FK matching
   - Random: Fallback random sampling when ML models are not available

Also includes PartitionedDataset for efficient handling of large partitioned datasets.
"""

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from copy import copy as shallow_copy
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
from mostlyai.sdk._data.base import DataIdentifier, DataTable, NonContextRelation, Schema
from mostlyai.sdk._data.file.base import FileDataTable
from mostlyai.sdk._data.util.common import IS_NULL, NON_CONTEXT_COLUMN_INFIX

_LOG = logging.getLogger(__name__)


# =============================================================================
# GLOBAL HYPERPARAMETER DEFAULTS FOR ML-BASED FK MODELS
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
TOP_K = 20
TOP_P = 0.95


# =============================================================================
# PARTITIONED DATASET FOR EFFICIENT DATA HANDLING
# =============================================================================


class PartitionedDataset:
    """Cached wrapper for FileDataTable with slicing and random sampling capabilities."""

    def __init__(self, table: FileDataTable, max_cached_partitions: int = 1):
        self.table = table
        self.max_cached_partitions = max_cached_partitions
        self.partition_info = []

        # unlimited cache if max_cached_partitions is -1
        cache_maxsize = None if max_cached_partitions == -1 else max_cached_partitions
        self._load_partition_cached = lru_cache(maxsize=cache_maxsize)(self._load_partition_uncached)

        self._build_partition_index()

    def _build_partition_index(self):
        """Build partition index using table's dataset files."""
        current_total = 0
        for file in self.table.dataset.files:
            partition_size = self._get_row_count_fast(file)
            self.partition_info.append(
                {
                    "file": file,
                    "start_idx": current_total,
                    "end_idx": current_total + partition_size,
                    "size": partition_size,
                }
            )
            current_total += partition_size

    def __getitem__(self, key) -> pd.DataFrame:
        """Support slicing: dataset[start:end]"""
        if isinstance(key, slice):
            return self._slice_data(key.start or 0, key.stop or len(self))
        else:
            raise TypeError("Key must be slice")

    def random_sample(self, n_items: int) -> pd.DataFrame:
        """Randomly sample n_items from the dataset."""
        if n_items <= 0:
            return pd.DataFrame()

        selected_partitions = set()
        total_available = 0
        available_partitions = list(range(len(self.partition_info)))
        np.random.shuffle(available_partitions)

        for partition_idx in available_partitions:
            if total_available >= n_items:
                break
            selected_partitions.add(partition_idx)
            total_available += self.partition_info[partition_idx]["size"]

        all_data = []
        for partition_idx in selected_partitions:
            partition = self.partition_info[partition_idx]
            df = self._load_partition(partition["file"])
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        if len(combined_df) >= n_items:
            sampled_df = combined_df.sample(n=n_items, replace=False).reset_index(drop=True)
        else:
            sampled_df = combined_df.sample(n=n_items, replace=True).reset_index(drop=True)

        return sampled_df

    def __len__(self) -> int:
        return self.table.row_count

    def _get_row_count_fast(self, file_path: str) -> int:
        """Get row count from parquet metadata without reading data."""
        if len(self.table.dataset.files) == 1:
            return self.table.row_count

        filesystem = self.table.container.file_system
        parquet_file = pq.ParquetFile(file_path, filesystem=filesystem)
        return parquet_file.metadata.num_rows

    def _load_partition_uncached(self, file_path: str) -> pd.DataFrame:
        """Load partition data from disk or remote storage (no caching)."""
        filesystem = self.table.container.file_system
        return pd.read_parquet(file_path, filesystem=filesystem)

    def _load_partition(self, file_path: str) -> pd.DataFrame:
        """Load partition with caching."""
        return self._load_partition_cached(file_path).copy()

    def _find_partition_for_index(self, global_idx: int) -> dict:
        """Find which partition contains the given global index."""
        for partition in self.partition_info:
            if partition["start_idx"] <= global_idx < partition["end_idx"]:
                return partition
        raise IndexError(f"Index {global_idx} out of range [0, {len(self)}")

    def _slice_data(self, start: int, end: int) -> pd.DataFrame:
        """Load data for slice range [start:end]"""
        if start >= len(self) or end <= 0:
            return pd.DataFrame()

        start = max(0, start)
        end = min(len(self), end)

        needed_partitions = []
        for partition in self.partition_info:
            if partition["end_idx"] > start and partition["start_idx"] < end:
                local_start = max(0, start - partition["start_idx"])
                local_end = min(partition["size"], end - partition["start_idx"])
                needed_partitions.append((partition, local_start, local_end))

        result_dfs = []
        for partition, local_start, local_end in needed_partitions:
            df = self._load_partition(partition["file"])
            slice_df = df.iloc[local_start:local_end]
            result_dfs.append(slice_df)

        return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()

    def iter_partitions(self) -> Iterator[tuple[int, str, pd.DataFrame]]:
        """Iterate over partitions yielding (index, file_path, dataframe)."""
        for idx, partition in enumerate(self.partition_info):
            file_path = partition["file"]
            data = self._load_partition(file_path)
            yield idx, file_path, data

    @property
    def files(self) -> list[str]:
        """Get partition file paths."""
        return self.table.dataset.files

    def clear_cache(self) -> None:
        """Clear the partition cache."""
        self._load_partition_cached.cache_clear()


# =============================================================================
# PULL PHASE: MARK NON-CONTEXT FKS FOR NULL HANDLING
# =============================================================================


def add_is_null_for_non_context_relations(
    schema: Schema,
    table_name: str,
    data: pd.DataFrame,
    is_target: bool,
) -> pd.DataFrame:
    """Handle all non-context relations for a table."""
    non_context_relations = schema.subset(
        relation_type=NonContextRelation,
        relations_to=[table_name],
    ).relations
    for relation in non_context_relations:
        data = add_is_null_for_non_context_relation(
            data=data,
            table=schema.tables[relation.parent.table],
            relation=relation,
            is_target=is_target,
        )
    return data


def add_is_null_for_non_context_relation(
    data: pd.DataFrame,
    table: DataTable,
    relation: NonContextRelation,
    is_target: bool = False,
) -> pd.DataFrame:
    """Handle a single non-context relation for a table and add an is_null column."""
    _LOG.info(f"handle non-context relation {table.name}")

    assert isinstance(relation, NonContextRelation)
    fk = relation.child.ref_name(prefixed=not is_target)
    if fk not in data:
        return data  # nothing to handle

    # identify which values in the FK column have no corresponding entry in the non-context table
    keys = set(data[fk].dropna())
    if len(keys) > 0:
        pk = table.primary_key
        pk_qual_name = DataIdentifier(table.name, pk).ref_name()
        # check for keys not in the parent table
        missing_keys = keys - set(
            table.read_data_prefixed(
                where={pk: list(keys)},
                columns=[pk],
                do_coerce_dtypes=True,
            )[pk_qual_name]
        )
    else:
        missing_keys = set()

    # create the is_null column based on whether a non-context foreign-key is present or not
    is_null_values = data[fk].apply(lambda x: str(pd.isna(x) or x in missing_keys))

    # replace the fk column with the is_null values and rename it accordingly
    data[fk] = is_null_values
    data.rename(columns={fk: relation.get_is_null_column(is_target=is_target)}, inplace=True)

    return data


# =============================================================================
# RANDOM FK ASSIGNMENT (FALLBACK)
# =============================================================================


def sample_non_context_keys(
    tgt_is_null: pd.Series,
    non_ctx_pks: pd.DataFrame,
) -> pd.Series:
    """
    Non-context matching algorithm. For each row in tgt_data, we randomly match a record in non_ctx_data.
    Returns pd.Series of sampled row indexes.
    """
    tgt_is_null = tgt_is_null.astype("string")
    # initialize returned pd.Series with NAs
    pk_dtype = non_ctx_pks.convert_dtypes(dtype_backend="pyarrow").dtype
    sampled_keys = pd.Series([pd.NA] * len(tgt_is_null), dtype=pk_dtype, index=tgt_is_null.index)
    # return immediately if no candidates to sample from
    if len(tgt_is_null) == 0:
        return sampled_keys
    tgt_to_sample = tgt_is_null[tgt_is_null != "True"].index
    samples = non_ctx_pks.sample(n=len(tgt_to_sample), replace=True).reset_index(drop=True)
    sampled_keys[tgt_to_sample] = samples
    return sampled_keys


def assign_non_context_fks_randomly(
    tgt_data: pd.DataFrame,
    generated_data_schema: Schema,
    tgt: str,
) -> pd.DataFrame:
    """
    Apply non-context keys allocation for each non-context relation for a generated table.
    Uses random sampling as a fallback when ML models are not available.
    """
    tgt_data = shallow_copy(tgt_data)
    for rel in generated_data_schema.relations:
        if not isinstance(rel, NonContextRelation) or rel.child.table != tgt:
            continue
        tgt_fk_name = rel.child.column
        tgt_is_null_column_name = rel.get_is_null_column()
        _LOG.info(f"sample non-context keys for {tgt_fk_name}")
        tgt_is_null = tgt_data[tgt_is_null_column_name]
        # read referenced table's keys
        non_ctx_pk_name = rel.parent.column
        non_ctx_pks = generated_data_schema.tables[rel.parent.table].read_data(
            do_coerce_dtypes=True, columns=[non_ctx_pk_name]
        )[non_ctx_pk_name]
        # sample non-ctx keys
        sampled_keys = sample_non_context_keys(tgt_is_null, non_ctx_pks)
        # replace is_null column with sampled keys
        tgt_data.insert(tgt_data.columns.get_loc(tgt_is_null_column_name), tgt_fk_name, sampled_keys)
        tgt_data = tgt_data.drop(columns=[tgt_is_null_column_name])
    return tgt_data


# =============================================================================
# ML-BASED FK MODELS: NEURAL NETWORK ARCHITECTURE
# =============================================================================


class EntityEncoder(nn.Module):
    """Neural network encoder for entity embeddings."""

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
    """Neural network model for parent-child relationship matching."""

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

        similarity = F.cosine_similarity(parent_encoded, child_encoded, dim=1)
        probability = torch.sigmoid(similarity * PEAKEDNESS_SCALER).unsqueeze(1)

        return probability


# =============================================================================
# ML-BASED FK MODELS: DATA ENCODING & STATISTICS
# =============================================================================


def safe_name(text: str) -> str:
    """Generate a safe filename with hash suffix."""
    safe = sanitize_filename(text)
    digest = hashlib.md5(safe.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def get_cardinalities(*, stats_dir: Path) -> dict[str, int]:
    """Extract cardinalities from stats file."""
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
    """Analyze dataframe and compute statistics for encoding."""
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

    stats_path = stats_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=4))


def encode_df(
    *, df: pd.DataFrame, stats_dir: Path, include_primary_key: bool = True, include_parent_key: bool = True
) -> pd.DataFrame:
    """Encode dataframe using pre-computed statistics."""
    stats_path = stats_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    primary_key = stats["primary_key"]
    parent_key = stats["parent_key"]
    cat_columns = stats["cat_columns"]
    num_columns = stats["num_columns"]
    dt_columns = stats["dt_columns"]

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


# =============================================================================
# ML-BASED FK MODELS: TRAINING DATA PREPARATION
# =============================================================================


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
        chunk_df = chunk_df.drop_duplicates(subset=[primary_key])

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
    parent_counts = defaultdict(int)
    collected_rows = []
    where = {tgt_parent_key: parent_keys}

    for chunk_df in tgt_table.read_chunks(where=where, columns=tgt_table.columns, do_coerce_dtypes=True):
        if len(chunk_df) == 0:
            continue

        for _, row in chunk_df.iterrows():
            parent_id = row[tgt_parent_key]

            if parent_counts[parent_id] < max_tgt_per_parent:
                collected_rows.append(row)
                parent_counts[parent_id] += 1

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
        tgt_table: Target/child table
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

    sample_size = min(int(sample_size), n_children)
    rng = np.random.default_rng()
    sampled_child_indices = rng.choice(n_children, size=sample_size, replace=False)
    children_X = children_X[sampled_child_indices]
    child_keys = child_keys[sampled_child_indices]

    # null children excluded from training - handled via _is_null column during inference
    non_null_mask = ~pd.isna(child_keys)
    children_X = children_X[non_null_mask]
    child_keys = child_keys[non_null_mask]
    n_non_null = len(children_X)

    if n_non_null == 0:
        raise ValueError("No non-null children found in training data")

    true_parent_pos = parent_index_by_key.loc[child_keys].to_numpy()
    if np.any(pd.isna(true_parent_pos)):
        raise ValueError("Some child foreign keys do not match any parent primary key")

    # positive pairs (label=1) - one per non-null child
    pos_parents = parents_X[true_parent_pos]
    pos_labels = np.ones(n_non_null, dtype=np.float32)

    # negative pairs (label=0) - n_negative per non-null child (vectorized)
    neg_indices = rng.integers(0, n_parents, size=(n_non_null, n_negative))

    # ensure negatives are not the true parent if there is more than one parent
    true_parent_pos_expanded = true_parent_pos[:, np.newaxis]
    mask = neg_indices == true_parent_pos_expanded

    while mask.any() and n_parents > 1:
        neg_indices[mask] = rng.integers(0, n_parents, size=mask.sum())
        mask = neg_indices == true_parent_pos_expanded

    neg_parents = parents_X[neg_indices.ravel()]
    neg_children = np.repeat(children_X, n_negative, axis=0)
    neg_labels = np.zeros(n_non_null * n_negative, dtype=np.float32)

    parent_vecs = np.vstack([pos_parents, neg_parents]).astype(np.float32, copy=False)
    child_vecs = np.vstack([children_X, neg_children]).astype(np.float32, copy=False)
    labels_vec = np.concatenate([pos_labels, neg_labels]).astype(np.float32, copy=False)

    parent_pd = pd.DataFrame(parent_vecs, columns=parent_encoded_data.drop(columns=[parent_primary_key]).columns)
    child_pd = pd.DataFrame(child_vecs, columns=tgt_encoded_data.drop(columns=[tgt_parent_key]).columns)
    labels_pd = pd.Series(labels_vec, name="labels")

    return parent_pd, child_pd, labels_pd


# =============================================================================
# ML-BASED FK MODELS: TRAINING
# =============================================================================


def train(
    *,
    model: ParentChildMatcher,
    parent_pd: pd.DataFrame,
    child_pd: pd.DataFrame,
    labels: pd.Series,
) -> None:
    """Train the parent-child matching model."""
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

        if val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                _LOG.info(f"Early stopping at epoch {epoch + 1}")
                break

    assert best_model_state is not None
    model.load_state_dict(best_model_state)
    _LOG.info("Best model restored (lowest validation loss).")


# =============================================================================
# ML-BASED FK MODELS: PERSISTENCE
# =============================================================================


def store_fk_model(*, model: ParentChildMatcher, fk_model_workspace_dir: Path) -> None:
    """Save FK model to disk."""
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
    """Load FK model from disk."""
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


# =============================================================================
# ML-BASED FK MODELS: INFERENCE
# =============================================================================


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
        child_embeddings = model.child_encoder(tgt_inputs)
        parent_embeddings = model.parent_encoder(parent_inputs)

        # create cartesian product: each child with all parent candidates
        child_embeddings_interleaved = child_embeddings.repeat_interleave(n_parent_batch, dim=0)
        parent_embeddings_interleaved = parent_embeddings.repeat(n_tgt, 1)

        similarity = F.cosine_similarity(parent_embeddings_interleaved, child_embeddings_interleaved, dim=1)
        similarity = similarity.view(n_tgt, n_parent_batch)
        prob_matrix = F.softmax(similarity * PEAKEDNESS_SCALER, dim=1)

        return prob_matrix


def sample_best_parents(
    *,
    prob_matrix: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
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
        top_p: If specified, use nucleus sampling - only sample from the smallest set
               of parents whose cumulative probability exceeds p (0.0 < p <= 1.0).
               This dynamically adjusts the candidate pool size based on probability mass.
               If both top_k and top_p are specified, top_k is applied first, then top_p.
               Recommended: 0.9-0.95 for high quality matches with adaptive diversity.

    Returns:
        best_parent_indices: Array of parent indices for each child
    """
    n_tgt = prob_matrix.shape[0]
    best_parent_indices = np.full(n_tgt, -1, dtype=np.int64)

    rng = np.random.default_rng()

    for i in range(n_tgt):
        if temperature == 0.0:
            best_parent_indices[i] = torch.argmax(prob_matrix[i]).cpu().numpy()
        else:
            probs = prob_matrix[i]
            candidate_indices = torch.arange(len(probs))

            # apply top_k filtering first if specified
            if top_k is not None and top_k < len(probs):
                top_k_values, top_k_indices = torch.topk(probs, k=top_k)
                probs = top_k_values
                candidate_indices = top_k_indices

            # apply top_p (nucleus) filtering if specified
            if top_p is not None and 0.0 < top_p < 1.0:
                # sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # compute cumulative probabilities
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)

                # find the cutoff index where cumulative probability exceeds top_p
                # keep at least one candidate
                cutoff_idx = torch.searchsorted(cumsum_probs, top_p, right=False).item() + 1
                cutoff_idx = max(1, min(cutoff_idx, len(sorted_probs)))

                # filter to nucleus candidates
                probs = sorted_probs[:cutoff_idx]
                candidate_indices = candidate_indices[sorted_indices[:cutoff_idx]]

            # apply temperature scaling (higher = more uniform sampling)
            logits = torch.log(probs + NUMERICAL_STABILITY_EPSILON) / temperature
            probs = torch.softmax(logits, dim=0).cpu().numpy()

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
    top_k: int | None = TOP_K,
    top_p: float | None = TOP_P,
) -> pd.DataFrame:
    """
    Match non-context foreign keys using trained ML models.

    This function uses a trained neural network to intelligently assign foreign keys
    based on the similarity between parent and child records.

    Args:
        fk_models_workspace_dir: Directory containing trained FK models
        tgt_data: Target/child data to assign FKs to
        parent_data: Parent data to sample from
        tgt_parent_key: Foreign key column name in target table
        parent_primary_key: Primary key column name in parent table
        parent_table_name: Name of parent table
        temperature: Sampling temperature (0=greedy, 1=normal, >1=diverse)
        top_k: Number of top candidates to consider per match
        top_p: Nucleus sampling threshold (0.0 < p <= 1.0) for dynamic candidate filtering

    Returns:
        Target data with FK column populated
    """
    # check for _is_null column (format: {fk_name}.{parent_table_name}._is_null)
    is_null_col = NON_CONTEXT_COLUMN_INFIX.join([tgt_parent_key, parent_table_name, IS_NULL])
    has_is_null = is_null_col in tgt_data.columns

    tgt_data[tgt_parent_key] = pd.NA

    if has_is_null:
        # _is_null column contains string values "True" or "False"
        is_null_values = tgt_data[is_null_col].astype(str)
        null_mask = is_null_values == "True"
        non_null_mask = ~null_mask

        _LOG.info(
            f"FK matching data | total_rows: {len(tgt_data)} | null_rows: {null_mask.sum()} | non_null_rows: {non_null_mask.sum()}"
        )

        if non_null_mask.sum() == 0:
            _LOG.warning(f"All rows have null FK values (via {is_null_col})")
            if is_null_col in tgt_data.columns:
                tgt_data = tgt_data.drop(columns=[is_null_col])
            return tgt_data

        non_null_indices = tgt_data.index[non_null_mask].tolist()

        tgt_data_non_null = tgt_data.loc[non_null_mask].copy().reset_index(drop=True)

        # remove _is_null column before encoding (not used by FK model)
        if is_null_col in tgt_data_non_null.columns:
            tgt_data_non_null = tgt_data_non_null.drop(columns=[is_null_col])
    else:
        _LOG.info(f"FK matching data | total_rows: {len(tgt_data)} | null_rows: 0 | non_null_rows: {len(tgt_data)}")
        tgt_data_non_null = tgt_data.copy()
        non_null_indices = tgt_data.index.tolist()
        non_null_mask = pd.Series(True, index=tgt_data.index)

    fk_model_workspace_dir = fk_models_workspace_dir / safe_name(tgt_parent_key)
    tgt_stats_dir = fk_model_workspace_dir / "tgt-stats"
    parent_stats_dir = fk_model_workspace_dir / "parent-stats"

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

    fk_parent_sample_size = len(parent_encoded)
    _LOG.info(
        f"FK model matching | temperature: {temperature} | top_k: {top_k} | top_p: {top_p} | parent_sample_size: {fk_parent_sample_size}"
    )

    prob_matrix = build_parent_child_probabilities(
        model=model,
        tgt_encoded=tgt_encoded,
        parent_encoded=parent_encoded,
    )

    best_parent_indices = sample_best_parents(
        prob_matrix=prob_matrix,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    best_parent_ids = parent_data.iloc[best_parent_indices][parent_primary_key].values

    parent_ids_series = pd.Series(best_parent_ids, index=non_null_indices)

    tgt_data.loc[non_null_indices, tgt_parent_key] = parent_ids_series

    if has_is_null and is_null_col in tgt_data.columns:
        tgt_data = tgt_data.drop(columns=[is_null_col])

    n_matched = non_null_mask.sum()
    n_null = (~non_null_mask).sum()
    _LOG.info(f"FK matching completed | matched: {n_matched} | null: {n_null}")

    return tgt_data
