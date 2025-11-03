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
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from copy import copy as shallow_copy
from copy import deepcopy
from functools import partial
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
from mostlyai.engine.domain import ModelEncodingType
from mostlyai.sdk._data.base import DataIdentifier, DataTable, NonContextRelation, Schema
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
PATIENCE = 5
N_NEGATIVE_SAMPLES = 5
VAL_SPLIT = 0.2
DROPOUT_RATE = 0.2
EARLY_STOPPING_DELTA = 1e-5
NUMERICAL_STABILITY_EPSILON = 1e-10

# Data Sampling Parameters
MAX_PARENT_SAMPLE_SIZE = 10000
MAX_TGT_PER_PARENT = 10

# Inference Parameters
TEMPERATURE = 1.0
TOP_K = None
TOP_P = 0.95
QUOTA_PENALTY_FACTOR = 0.1

# Supported Encoding Types
FK_MODEL_ENCODING_TYPES = [
    ModelEncodingType.tabular_categorical,
    ModelEncodingType.tabular_numeric_auto,
    ModelEncodingType.tabular_numeric_discrete,
    ModelEncodingType.tabular_numeric_binned,
    ModelEncodingType.tabular_numeric_digit,
    ModelEncodingType.tabular_datetime,
]


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

        logits = F.cosine_similarity(parent_encoded, child_encoded, dim=1)
        logits = (logits * PEAKEDNESS_SCALER).unsqueeze(1)

        return logits


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
            analyze, reduce = (
                partial(analyze_numeric, encoding_type=ModelEncodingType.tabular_numeric_digit),
                partial(analyze_reduce_numeric, encoding_type=ModelEncodingType.tabular_numeric_digit),
            )
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
    t0 = time.time()
    primary_key = parent_table.primary_key
    assert primary_key is not None
    foreign_keys = [fk.column for fk in parent_table.foreign_keys]
    data_columns = [
        c
        for c in parent_table.columns
        if c != primary_key  # data column is not the primary key
        and c not in foreign_keys  # data column is not a foreign key
        and parent_table.encoding_types.get(c) in FK_MODEL_ENCODING_TYPES  # encoding type is supported by FK models
    ]
    columns = [primary_key] + data_columns
    seen_keys = set()
    collected_rows = []

    for chunk_df in parent_table.read_chunks(columns=columns, do_coerce_dtypes=True):
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
    _LOG.info(f"fetch_parent_data | fetched: {len(parent_data)} | time: {time.time() - t0:.2f}s")
    return parent_data


def fetch_tgt_data(
    *,
    tgt_table: DataTable,
    tgt_parent_key: str,
    parent_keys: list,
    schema: Schema,
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
        schema: Schema to fetch context parent data for tgt_table
        max_tgt_per_parent: Maximum target records per parent.

    Returns:
        DataFrame containing target records, limited by max_tgt_per_parent constraint.
    """
    t0 = time.time()
    parent_counts = defaultdict(int)
    collected_rows = []
    where = {tgt_parent_key: parent_keys}
    tgt_primary_key = tgt_table.primary_key
    tgt_context_key = schema.get_context_key(tgt_table.name)
    tgt_foreign_keys = [fk.column for fk in tgt_table.foreign_keys]
    data_columns = [
        c
        for c in tgt_table.columns
        if c != tgt_primary_key  # data column is not the primary key
        and c not in tgt_foreign_keys  # data column is not a foreign key
        and tgt_table.encoding_types.get(c) in FK_MODEL_ENCODING_TYPES  # encoding type is supported by FK models
    ]
    columns = [tgt_parent_key]
    if tgt_context_key is not None:
        columns += [tgt_context_key.column]
    columns += data_columns

    for chunk_df in tgt_table.read_chunks(columns=columns, where=where, do_coerce_dtypes=True):
        if len(chunk_df) == 0:
            continue

        for _, row in chunk_df.iterrows():
            parent_id = row[tgt_parent_key]

            if parent_counts[parent_id] < max_tgt_per_parent:
                collected_rows.append(row)
                parent_counts[parent_id] += 1

    tgt_data = pd.DataFrame(collected_rows).reset_index(drop=True)
    _LOG.info(f"fetch_tgt_data | fetched: {len(tgt_data)} | time: {time.time() - t0:.2f}s")
    return tgt_data


def add_context_parent_data(
    *,
    tgt_data: pd.DataFrame,
    tgt_table: DataTable,
    schema: Schema,
    drop_context_key: bool = False,
) -> pd.DataFrame:
    t0 = time.time()

    ctx_relation = schema.get_parent_context_relation(tgt_table.name)
    if ctx_relation is None:
        # no context parent, return as is
        return tgt_data

    ctx_parent_table_name = ctx_relation.parent.table
    ctx_parent_table = schema.tables[ctx_parent_table_name]
    ctx_parent_pk = ctx_relation.parent.column
    tgt_ctx_fk = ctx_relation.child.column

    # get unique context parent keys from tgt_data
    ctx_parent_keys = tgt_data[tgt_ctx_fk].dropna().unique().tolist()
    if not ctx_parent_keys:
        _LOG.info(f"No context parent keys found in tgt_data for {tgt_table.name}")
        return tgt_data

    # identify key columns to exclude (primary key + foreign keys)
    key_columns = {ctx_parent_pk}
    for rel in schema.relations:
        if rel.child.table == ctx_parent_table_name:
            key_columns.add(rel.child.column)

    # get non-key columns that are supported by FK models
    include_columns = []
    for col in ctx_parent_table.columns:
        if col in key_columns:
            continue
        # only include columns with encoding types supported by FK models
        encoding_type = ctx_parent_table.encoding_types.get(col)
        if encoding_type not in FK_MODEL_ENCODING_TYPES:
            continue
        include_columns.append(col)

    # fetch context parent data with prefixed columns (only non-key columns + PK for join)
    columns_to_fetch = [ctx_parent_pk] + include_columns
    ctx_parent_data = ctx_parent_table.read_data_prefixed(
        columns=columns_to_fetch,
        where={ctx_parent_pk: ctx_parent_keys},
        do_coerce_dtypes=True,
    )

    if ctx_parent_data.empty:
        _LOG.info(f"No context parent data found for {tgt_table.name}")
        return tgt_data

    # join context parent data with tgt_data
    ctx_parent_pk_prefixed = DataIdentifier(ctx_parent_table.name, ctx_parent_pk).ref_name()
    tgt_data = pd.merge(
        tgt_data,
        ctx_parent_data,
        left_on=tgt_ctx_fk,
        right_on=ctx_parent_pk_prefixed,
        how="left",
    )

    # drop the primary key column after join (only keep non-key columns)
    if ctx_parent_pk_prefixed in tgt_data.columns:
        tgt_data = tgt_data.drop(columns=[ctx_parent_pk_prefixed])

    # drop the context key column if requested
    if drop_context_key:
        tgt_data = tgt_data.drop(columns=[tgt_ctx_fk])

    added_columns = [c for c in tgt_data.columns if c in ctx_parent_data.columns]
    _LOG.info(f"add_context_parent_data | time: {time.time() - t0:.2f}s | added_columns: {added_columns}")
    return tgt_data


def pull_fk_model_training_data(
    *,
    tgt_table: DataTable,
    parent_table: DataTable,
    tgt_parent_key: str,
    schema: Schema,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull training data for a specific non-context FK relation.

    Args:
        tgt_table: Target/child table
        parent_table: Parent table
        tgt_parent_key: Foreign key column in target table
        schema: Schema to fetch context parent data for tgt_table

    Returns:
        Tuple of (parent_data, tgt_data). tgt_data includes columns from context parent table.
    """
    parent_data = fetch_parent_data(parent_table=parent_table)

    parent_keys = parent_data[parent_table.primary_key].tolist() if not parent_data.empty else []
    tgt_data = fetch_tgt_data(
        tgt_table=tgt_table,
        tgt_parent_key=tgt_parent_key,
        parent_keys=parent_keys,
        schema=schema,
    )
    tgt_data = add_context_parent_data(
        tgt_data=tgt_data,
        tgt_table=tgt_table,
        schema=schema,
        drop_context_key=True,  # after this step, tgt_context_key is dropped
    )
    _LOG.info(
        f"pull_fk_model_training_data | parent_data columns: {list(parent_data.columns)} | tgt_data columns: {list(tgt_data.columns)}"
    )
    return parent_data, tgt_data


def prepare_training_pairs(
    parent_encoded_data: pd.DataFrame,
    tgt_encoded_data: pd.DataFrame,
    parent_primary_key: str,
    tgt_parent_key: str,
    n_negative_samples: int = N_NEGATIVE_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepare training data for a parent-child matching model.
    For each non-null child, samples will include:
    - One unique positive pair (correct parent) with label=1
    - Multiple negative pairs (wrong parents) with label=0

    Class imbalance is handled at the loss level using BCEWithLogitsLoss with pos_weight.

    Args:
        parent_encoded_data: Encoded parent data
        tgt_encoded_data: Encoded child data
        parent_primary_key: Primary key of parents
        tgt_parent_key: Foreign key of children
        n_negative_samples: Number of negative samples per child
    """
    t0 = time.time()

    parent_keys = parent_encoded_data[parent_primary_key].to_numpy()
    parents_X = parent_encoded_data.drop(columns=[parent_primary_key]).to_numpy(dtype=np.float32)
    n_parents = parents_X.shape[0]
    parent_index_by_key = pd.Series(np.arange(n_parents), index=parent_keys)

    tgt_keys = tgt_encoded_data[tgt_parent_key].to_numpy()
    tgt_X = tgt_encoded_data.drop(columns=[tgt_parent_key]).to_numpy(dtype=np.float32)
    n_tgt = tgt_X.shape[0]

    # shuffle tgt keys
    shuffled = np.random.permutation(n_tgt)
    tgt_X = tgt_X[shuffled]
    tgt_keys = tgt_keys[shuffled]

    # exclude null tgt keys from training
    non_null_mask = ~pd.isna(tgt_keys)
    tgt_X = tgt_X[non_null_mask]
    tgt_keys = tgt_keys[non_null_mask]
    n_non_null = len(tgt_X)

    if n_non_null == 0:
        raise ValueError("No non-null children found in training data")

    # exclude tgt keys that don't match any parent
    valid_parent_mask = pd.Series(tgt_keys).isin(parent_keys).to_numpy()
    if not valid_parent_mask.all():
        n_invalid = (~valid_parent_mask).sum()
        _LOG.warning(f"Dropping {n_invalid} child records with foreign keys not matching any parent primary key")
    tgt_X = tgt_X[valid_parent_mask]
    tgt_keys = tgt_keys[valid_parent_mask]
    n_valid = len(tgt_X)

    if n_valid == 0:
        raise ValueError("No valid children found in training data (all foreign keys are null or invalid)")

    true_parent_pos = parent_index_by_key.loc[tgt_keys].to_numpy()

    # positive pairs (label=1) - one per valid tgt
    pos_parents = parents_X[true_parent_pos]
    pos_labels = np.ones(n_valid, dtype=np.float32)

    # negative pairs (label=0) - n_negative_samples per valid tgt
    neg_indices = np.random.randint(0, n_parents, size=(n_valid, n_negative_samples))
    if n_parents > 1:
        true_parent_pos_expanded = true_parent_pos[:, np.newaxis]
        mask = neg_indices == true_parent_pos_expanded
        while mask.any():
            neg_indices[mask] = np.random.randint(0, n_parents, size=mask.sum())
            mask = neg_indices == true_parent_pos_expanded
    neg_parents = parents_X[neg_indices.ravel()]
    neg_tgt = np.repeat(tgt_X, n_negative_samples, axis=0)
    neg_labels = np.zeros(n_valid * n_negative_samples, dtype=np.float32)

    parent_vecs = np.vstack([pos_parents, neg_parents]).astype(np.float32, copy=False)
    tgt_vecs = np.vstack([tgt_X, neg_tgt]).astype(np.float32, copy=False)
    labels_vec = np.concatenate([pos_labels, neg_labels]).astype(np.float32, copy=False)

    # shuffle pairs for training robustness
    n_pairs = len(parent_vecs)
    shuffle_indices = np.random.permutation(n_pairs)
    parent_vecs = parent_vecs[shuffle_indices]
    tgt_vecs = tgt_vecs[shuffle_indices]
    labels_vec = labels_vec[shuffle_indices]

    parent_pd = pd.DataFrame(parent_vecs, columns=parent_encoded_data.drop(columns=[parent_primary_key]).columns)
    tgt_pd = pd.DataFrame(tgt_vecs, columns=tgt_encoded_data.drop(columns=[tgt_parent_key]).columns)
    labels_pd = pd.Series(labels_vec, name="labels")

    n_pairs = len(parent_pd)
    _LOG.info(f"prepare_training_pairs | n_pairs: {n_pairs} | time: {time.time() - t0:.2f}s")
    return parent_pd, tgt_pd, labels_pd


# =============================================================================
# ML-BASED FK MODELS: TRAINING
# =============================================================================


def train_fk_model(
    *,
    model: ParentChildMatcher,
    parent_pd: pd.DataFrame,
    tgt_pd: pd.DataFrame,
    labels: pd.Series,
) -> None:
    """Train the parent-child matching model."""
    t0 = time.time()

    X_parent = torch.tensor(parent_pd.values, dtype=torch.int64)
    X_tgt = torch.tensor(tgt_pd.values, dtype=torch.int64)
    y = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_parent, X_tgt, y)

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # calculate class imbalance for loss weighting
    num_positives = int(labels.sum())
    num_negatives = len(labels) - num_positives
    pos_weight = torch.tensor([num_negatives / num_positives]) if num_positives > 0 else torch.tensor([1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, val_losses = [], []
    best_model_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_parent, batch_tgt, batch_y in train_loader:
            batch_parent = {col: batch_parent[:, i] for i, col in enumerate(parent_pd.columns)}
            batch_tgt = {col: batch_tgt[:, i] for i, col in enumerate(tgt_pd.columns)}
            optimizer.zero_grad()
            pred = model(batch_parent, batch_tgt)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_y.size(0)
        train_loss /= train_size
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_parent, batch_tgt, batch_y in val_loader:
                batch_parent = {col: batch_parent[:, i] for i, col in enumerate(parent_pd.columns)}
                batch_tgt = {col: batch_tgt[:, i] for i, col in enumerate(tgt_pd.columns)}
                pred = model(batch_parent, batch_tgt)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_y.size(0)
        val_loss /= val_size
        val_losses.append(val_loss)

        progress_msg = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
        }
        _LOG.info(progress_msg)

        if val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            epochs_no_improve = 0
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                _LOG.info("early stopping: val_loss stopped improving")
                break

    assert best_model_state is not None
    model.load_state_dict(best_model_state)
    _LOG.info(
        f"train_fk_model() | time: {time.time() - t0:.2f}s | best_epoch: {best_epoch} | best_val_loss: {best_val_loss:.4f}"
    )


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
    parent_ids: list,
    remaining_capacity: dict | None = None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> np.ndarray:
    """
    Sample best parent for each child based on match probabilities.

    Args:
        prob_matrix: (n_tgt, n_parent) probability each parent is a match
        parent_ids: List of parent IDs corresponding to columns in prob_matrix
        remaining_capacity: Optional mutable dict {parent_id: remaining_slots} for quota enforcement.
                           When provided, enables dynamic quota enforcement - capacity is decremented
                           during sampling to prevent parents from exceeding their target.
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
        Array of parent IDs for each child
    """
    n_tgt = prob_matrix.shape[0]

    if remaining_capacity is not None:
        # Dynamic quota enforcement - iterative sampling with capacity updates
        available_mask = np.array([remaining_capacity.get(pid, 0) > 0 for pid in parent_ids], dtype=bool)
        assigned_parent_ids = []

        for child_idx in range(n_tgt):
            # Get row probabilities for this child
            row_probs = prob_matrix[child_idx].clone()

            # Apply penalty to parents at/over quota (instead of zeroing them out)
            row_probs[~torch.tensor(available_mask)] *= QUOTA_PENALTY_FACTOR

            # Renormalize
            row_sum = row_probs.sum()
            if row_sum > 0:
                row_probs = row_probs / row_sum
            else:
                # Fallback: all parents at quota, reset availability
                available_mask[:] = True
                row_probs = prob_matrix[child_idx].clone()
                row_probs = row_probs / row_probs.sum()

            # Sample parent for this child
            parent_idx = sample_single_parent(
                probs=row_probs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            parent_id = parent_ids[parent_idx]
            assigned_parent_ids.append(parent_id)

            # DECREMENT remaining capacity immediately
            remaining_capacity[parent_id] -= 1

            # Update mask if parent depleted
            if remaining_capacity[parent_id] <= 0:
                available_mask[parent_idx] = False

        return np.array(assigned_parent_ids)
    else:
        # Batch sampling (no quota enforcement)
        best_parent_indices = []

        for i in range(n_tgt):
            parent_idx = sample_single_parent(
                probs=prob_matrix[i],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            best_parent_indices.append(parent_idx)

        # Convert indices to parent IDs
        return np.array([parent_ids[idx] for idx in best_parent_indices])


def sample_single_parent(
    *,
    probs: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> int:
    """
    Sample a single parent index based on match probabilities.

    Args:
        probs: 1D tensor of probabilities for parent candidates
        temperature: Controls variance in parent selection
        top_k: Only sample from top-K most probable parents
        top_p: Nucleus sampling threshold

    Returns:
        parent_index: Index of sampled parent
    """
    rng = np.random.default_rng()

    if temperature == 0.0:
        return torch.argmax(probs).cpu().item()

    candidate_indices = torch.arange(len(probs))

    # apply top_k filtering first if specified
    if top_k is not None and top_k < len(probs):
        top_k_values, top_k_indices = torch.topk(probs, k=top_k)
        probs = top_k_values
        candidate_indices = top_k_indices

    # apply top_p (nucleus) filtering if specified
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_idx = torch.searchsorted(cumsum_probs, top_p, right=False).item() + 1
        cutoff_idx = max(1, min(cutoff_idx, len(sorted_probs)))
        probs = sorted_probs[:cutoff_idx]
        candidate_indices = candidate_indices[sorted_indices[:cutoff_idx]]

    # apply temperature scaling
    logits = torch.log(probs + NUMERICAL_STABILITY_EPSILON) / temperature
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    sampled_candidate = rng.choice(len(probs), p=probs)
    return candidate_indices[sampled_candidate].cpu().item()


def match_non_context(
    *,
    fk_models_workspace_dir: Path,
    tgt_data: pd.DataFrame,
    parent_data: pd.DataFrame,
    tgt_parent_key: str,
    parent_primary_key: str,
    parent_table_name: str,
    remaining_capacity: dict | None = None,
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
        remaining_capacity: Optional mutable dict {parent_id: remaining_slots} for quota enforcement.
                           When provided, enables dynamic quota enforcement - capacity is decremented
                           during sampling to prevent parents from exceeding their target.
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

    # Sample parents (with optional quota enforcement)
    parent_ids_list = parent_data[parent_primary_key].tolist()
    best_parent_ids = sample_best_parents(
        prob_matrix=prob_matrix,
        parent_ids=parent_ids_list,
        remaining_capacity=remaining_capacity,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    parent_ids_series = pd.Series(best_parent_ids, index=non_null_indices)

    tgt_data.loc[non_null_indices, tgt_parent_key] = parent_ids_series

    if has_is_null and is_null_col in tgt_data.columns:
        tgt_data = tgt_data.drop(columns=[is_null_col])

    n_matched = non_null_mask.sum()
    n_null = (~non_null_mask).sum()
    _LOG.info(f"FK matching completed | matched: {n_matched} | null: {n_null}")

    return tgt_data
