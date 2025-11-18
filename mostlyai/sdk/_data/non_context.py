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
import shutil
import time
from copy import copy as shallow_copy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathvalidate import sanitize_filename
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import mostlyai.engine as engine
from mostlyai.engine._common import read_json, write_json
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
MIN_ENTITY_EMBEDDING_DIM = 128
PEAKEDNESS_SCALER = 7.0

# Training Parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_MIN_LR = 1e-6
MAX_EPOCHS = 1000
PATIENCE = 10
N_NEGATIVE_SAMPLES = 20
VAL_SPLIT = 0.2
DROPOUT_RATE = 0.2
EARLY_STOPPING_DELTA = 1e-5
NUMERICAL_STABILITY_EPSILON = 1e-10


# Data Sampling Parameters
MAX_TGT_PER_PARENT = 10
MAX_CHILDREN = 5000

# Inference Parameters
TEMPERATURE = 1.0
TOP_K = None
TOP_P = 0.95
QUOTA_PENALTY_FACTOR = 0.02

# Supported Encoding Types for FK Models
FK_MODEL_ENCODING_TYPES = [
    ModelEncodingType.tabular_categorical,
    ModelEncodingType.tabular_numeric_auto,
    ModelEncodingType.tabular_numeric_discrete,
    ModelEncodingType.tabular_numeric_binned,
    ModelEncodingType.tabular_numeric_digit,
    ModelEncodingType.tabular_datetime,
]

# Column name for children count in cardinality models
CHILDREN_COUNT_COLUMN_NAME = "__CHILDREN_COUNT__"


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
    _LOG.info(f"[NonContext] Handle relation | table={table.name}")

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
        _LOG.info(f"[NonContext] Sample keys | fk={tgt_fk_name}")
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
        sub_column_embedding_dim: int,
        entity_hidden_dim: int,
        entity_embedding_dim: int,
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
            nn.Linear(entity_dim, self.entity_embedding_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.entity_embedding_dim, self.entity_hidden_dim),
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
        sub_column_embedding_dim: int | None = None,
        entity_hidden_dim: int | None = None,
        parent_entity_embedding_dim: int | None = None,
        child_entity_embedding_dim: int | None = None,
    ):
        super().__init__()

        sub_column_embedding_dim = sub_column_embedding_dim or SUB_COLUMN_EMBEDDING_DIM

        parent_entity_embedding_dim = parent_entity_embedding_dim or max(
            MIN_ENTITY_EMBEDDING_DIM, int((len(parent_cardinalities) * sub_column_embedding_dim) ** 0.5)
        )
        child_entity_embedding_dim = child_entity_embedding_dim or max(
            MIN_ENTITY_EMBEDDING_DIM, int((len(child_cardinalities) * sub_column_embedding_dim) ** 0.5)
        )

        entity_hidden_dim = entity_hidden_dim or (max(parent_entity_embedding_dim, child_entity_embedding_dim) * 2)

        self.parent_encoder = EntityEncoder(
            cardinalities=parent_cardinalities,
            sub_column_embedding_dim=sub_column_embedding_dim,
            entity_hidden_dim=entity_hidden_dim,
            entity_embedding_dim=parent_entity_embedding_dim,
        )
        self.child_encoder = EntityEncoder(
            cardinalities=child_cardinalities,
            sub_column_embedding_dim=sub_column_embedding_dim,
            entity_hidden_dim=entity_hidden_dim,
            entity_embedding_dim=child_entity_embedding_dim,
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
    encoding_types: dict[str, ModelEncodingType] | None = None,
    stats_dir: Path,
) -> None:
    """
    Analyze dataframe and compute statistics for encoding.

    Args:
        df: DataFrame to analyze
        primary_key: Primary key column name (excluded from analysis)
        parent_key: Parent key column name (excluded from analysis)
        data_columns: List of data columns to analyze
        encoding_types: Resolved encoding types (column_name -> ModelEncodingType).
                       When provided, these are used to determine column categories.
                       When None, infers categories from pandas dtypes.
        stats_dir: Directory to save statistics
    """
    stats_dir.mkdir(parents=True, exist_ok=True)

    key_columns = []
    if primary_key is not None:
        key_columns.append(primary_key)
    if parent_key is not None:
        key_columns.append(parent_key)

    data_columns = data_columns or list(df.columns)

    # preserve column order to ensure deterministic encoding
    data_columns = [col for col in data_columns if col not in key_columns and col in df.columns]

    # Determine column types: use user-defined encoding types if available, otherwise infer from dtypes
    if encoding_types:
        _LOG.info(f"[NonContext] Using user-defined encoding types | columns={list(encoding_types.keys())}")
        num_columns = []
        dt_columns = []
        cat_columns = []

        for col in data_columns:
            encoding_type = encoding_types.get(col)
            if encoding_type is None:
                # Column not in encoding_types, infer from pandas dtype
                if col in df.select_dtypes(include="number").columns:
                    num_columns.append(col)
                elif col in df.select_dtypes(include="datetime").columns:
                    dt_columns.append(col)
                else:
                    cat_columns.append(col)
                continue

            # Map numeric encoding types
            if encoding_type in [
                ModelEncodingType.tabular_numeric_auto,
                ModelEncodingType.tabular_numeric_discrete,
                ModelEncodingType.tabular_numeric_binned,
                ModelEncodingType.tabular_numeric_digit,
            ]:
                num_columns.append(col)
            # Map datetime encoding types
            elif encoding_type == ModelEncodingType.tabular_datetime:
                dt_columns.append(col)
            # Map categorical encoding type
            elif encoding_type == ModelEncodingType.tabular_categorical:
                cat_columns.append(col)

        _LOG.info(
            f"[NonContext] Column mapping | numeric={num_columns} | datetime={dt_columns} | categorical={cat_columns}"
        )
    else:
        # Fallback to dtype inference when no encoding types provided
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
    write_json(stats, stats_path)


def encode_df(
    *, df: pd.DataFrame, stats_dir: Path, include_primary_key: bool = True, include_parent_key: bool = True
) -> pd.DataFrame:
    """Encode dataframe using pre-computed statistics."""
    stats_path = stats_dir / "stats.json"
    stats = read_json(stats_path)
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
        _LOG.info(f"[NonContext] No parent keys found | table={tgt_table.name}")
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
        _LOG.info(f"[NonContext] No parent data found | table={tgt_table.name}")
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
    _LOG.info(f"[NonContext] Add context parent data | time={time.time() - t0:.2f}s | added_columns={added_columns}")
    return tgt_data


def pull_fk_model_training_data(
    *,
    tgt_table: DataTable,
    parent_table: DataTable,
    tgt_parent_key: str,
    schema: Schema,
    max_children_per_parent: int = MAX_TGT_PER_PARENT,
    max_children: int = MAX_CHILDREN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull training data for a specific non-context FK relation.

    Args:
        tgt_table: Target/child table
        parent_table: Parent table
        tgt_parent_key: Foreign key column in target table
        schema: Schema to fetch context parent data for tgt_table
        max_children_per_parent: Maximum children to keep per parent
        max_children: Maximum total children to keep for training

    Returns:
        Tuple of (parent_data, tgt_data). tgt_data includes columns from context parent table.
    """
    t0 = time.time()
    parent_primary_key = parent_table.primary_key
    assert parent_primary_key is not None

    # Pull ALL parent keys from children (we only need the FK column for filtering)
    all_children_keys = tgt_table.read_data(columns=[tgt_parent_key], do_coerce_dtypes=True)

    # Pull ALL parent keys
    all_parent_keys = (
        parent_table.read_data(columns=[parent_primary_key], do_coerce_dtypes=True)[parent_primary_key]
        .dropna()
        .unique()
    )
    all_parent_keys_set = set(all_parent_keys)

    # Filter children to max_children_per_parent
    # Remove rows where parent key is null or not in parent table
    all_children_keys = all_children_keys[
        all_children_keys[tgt_parent_key].notna() & all_children_keys[tgt_parent_key].isin(all_parent_keys_set)
    ]
    filtered_children_keys = all_children_keys.groupby(tgt_parent_key, as_index=False).head(max_children_per_parent)

    # Keep max_children for training
    if len(filtered_children_keys) > max_children:
        filtered_children_keys = filtered_children_keys.sample(n=max_children).reset_index(drop=True)

    # Identify parent_keys that have children in this subset
    parent_keys_with_children = set(filtered_children_keys[tgt_parent_key].unique())

    # Sample parent_keys without children
    parent_keys_without_children = all_parent_keys_set - parent_keys_with_children

    if len(parent_keys_without_children) > 0:
        # Sample same number as parents with children for balanced training
        n_to_sample = min(len(parent_keys_with_children), len(parent_keys_without_children))
        sampled_parent_keys_without_children = set(
            np.random.choice(list(parent_keys_without_children), size=n_to_sample, replace=False)
        )
    else:
        sampled_parent_keys_without_children = set()

    # Combine all parent keys to fetch
    final_parent_keys = list(parent_keys_with_children | sampled_parent_keys_without_children)

    # Fetch actual parent data
    parent_foreign_keys = [fk.column for fk in parent_table.foreign_keys]
    parent_data_columns = [
        c
        for c in parent_table.columns
        if c != parent_primary_key  # data column is not the primary key
        and c not in parent_foreign_keys  # data column is not a foreign key
        and parent_table.encoding_types.get(c) in FK_MODEL_ENCODING_TYPES  # encoding type is supported by FK models
    ]
    parent_columns = [parent_primary_key] + parent_data_columns
    parent_data = parent_table.read_data(
        columns=parent_columns,
        where={parent_primary_key: final_parent_keys},
        do_coerce_dtypes=True,
    )

    # Fetch actual children data using parent keys
    tgt_primary_key = tgt_table.primary_key
    tgt_context_key = schema.get_context_key(tgt_table.name)
    tgt_foreign_keys = [fk.column for fk in tgt_table.foreign_keys]
    tgt_data_columns = [
        c
        for c in tgt_table.columns
        if c != tgt_primary_key  # data column is not the primary key
        and c not in tgt_foreign_keys  # data column is not a foreign key
        and tgt_table.encoding_types.get(c) in FK_MODEL_ENCODING_TYPES  # encoding type is supported by FK models
    ]
    tgt_columns = [tgt_parent_key]
    if tgt_context_key is not None:
        tgt_columns.append(tgt_context_key.column)
    tgt_columns += tgt_data_columns

    # Fetch children WHERE parent_key IN (parents_with_children)
    tgt_data = tgt_table.read_data(
        columns=tgt_columns,
        where={tgt_parent_key: parent_keys_with_children},
        do_coerce_dtypes=True,
    )

    # Re-apply filtering: max children per parent, then max total children
    tgt_data = tgt_data.groupby(tgt_parent_key, as_index=False).head(max_children_per_parent)
    if len(tgt_data) > max_children:
        tgt_data = tgt_data.sample(n=max_children).reset_index(drop=True)

    # Add context parent data
    tgt_data = add_context_parent_data(
        tgt_data=tgt_data,
        tgt_table=tgt_table,
        schema=schema,
        drop_context_key=True,  # after this step, tgt_context_key is dropped
    )

    avg_children_per_parent = (
        len(tgt_data) / len(parent_keys_with_children) if len(parent_keys_with_children) > 0 else 0
    )
    _LOG.info(
        f"[NonContext] Pull training data | parents_total={len(parent_data)} | parents_with_children={len(parent_keys_with_children)} | parents_without={len(sampled_parent_keys_without_children)} | children={len(tgt_data)} | max_per_parent={max_children_per_parent} | avg_per_parent={avg_children_per_parent:.2f} | time={time.time() - t0:.2f}s"
    )
    return parent_data, tgt_data


def prepare_training_data_for_cardinality_model(
    *,
    parent_data: pd.DataFrame,
    tgt_data: pd.DataFrame,
    parent_primary_key: str,
    tgt_parent_key: str,
) -> pd.DataFrame:
    """
    Prepare parent data enriched with children count column for engine training.

    This function adds a special '__CHILDREN_COUNT__' column to the parent data that
    represents how many children each parent has. This enriched dataset is then used
    to train the mostlyai-engine model, which can later predict children counts for
    new synthetic parents.

    Args:
        parent_data: Parent table data (unencoded, raw features)
        tgt_data: Target/child table data (unencoded)
        parent_primary_key: Primary key column in parent data
        tgt_parent_key: Foreign key column in target data pointing to parent

    Returns:
        Parent data with added '__CHILDREN_COUNT__' column
    """
    t0 = time.time()

    children_counts = tgt_data[tgt_parent_key].value_counts()
    children_counts = parent_data[parent_primary_key].map(children_counts).fillna(0).astype(int)
    parent_data_enriched = parent_data.assign(**{CHILDREN_COUNT_COLUMN_NAME: children_counts})

    _LOG.info(
        f"[NonContext Cardinality] Prepare training data | n_rows={len(parent_data_enriched)} | time={time.time() - t0:.2f}s"
    )

    return parent_data_enriched


def prepare_training_pairs_for_fk_model(
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
    _LOG.info(f"[NonContext Matching] Prepare training pairs | n_pairs={n_pairs} | time={time.time() - t0:.2f}s")
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

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

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        progress_msg = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": f"{current_lr:.2e}",
        }
        _LOG.info(f"[NonContext Matching] {progress_msg}")

        if val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            epochs_no_improve = 0
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                _LOG.info("[NonContext Matching] Early stopping | reason=val_loss_plateau")
                break

    assert best_model_state is not None
    model.load_state_dict(best_model_state)
    _LOG.info(
        f"[NonContext Matching] Train FK model | time={time.time() - t0:.2f}s | best_epoch={best_epoch} | best_val_loss={best_val_loss:.4f}"
    )


# =============================================================================
# ML-BASED FK MODELS: PERSISTENCE
# =============================================================================


def store_fk_model(*, model: ParentChildMatcher, fk_model_workspace_dir: Path) -> None:
    """Save FK model to disk."""
    matching_model_dir = fk_model_workspace_dir / "fk_matching_model"
    matching_model_dir.mkdir(parents=True, exist_ok=True)
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
    }
    model_config_path = matching_model_dir / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=4))
    model_state_path = matching_model_dir / "model_weights.pt"
    torch.save(model.state_dict(), model_state_path)


def load_fk_model(*, fk_model_workspace_dir: Path) -> ParentChildMatcher:
    """Load FK model from disk."""
    matching_model_dir = fk_model_workspace_dir / "fk_matching_model"
    model_config_path = matching_model_dir / "model_config.json"
    model_config = json.loads(model_config_path.read_text())
    model = ParentChildMatcher(
        parent_cardinalities=model_config["parent_encoder"]["cardinalities"],
        child_cardinalities=model_config["child_encoder"]["cardinalities"],
        sub_column_embedding_dim=model_config["parent_encoder"]["sub_column_embedding_dim"],
        entity_hidden_dim=model_config["parent_encoder"]["entity_hidden_dim"],
        parent_entity_embedding_dim=model_config["parent_encoder"]["entity_embedding_dim"],
        child_entity_embedding_dim=model_config["child_encoder"]["entity_embedding_dim"],
    )
    model_state_path = matching_model_dir / "model_weights.pt"
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
    remaining_capacity: dict,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> np.ndarray:
    """
    Sample best parent for each child based on match probabilities.

    Args:
        prob_matrix: (n_tgt, n_parent) probability each parent is a match
        parent_ids: List of parent IDs corresponding to columns in prob_matrix
        remaining_capacity: dict {parent_id: remaining_slots} for quota enforcement.
                           Enables dynamic quota enforcement - capacity is decremented
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

    # Dynamic quota enforcement - iterative sampling with capacity updates
    available_mask = np.array([remaining_capacity.get(pid, 0) > 0 for pid in parent_ids], dtype=bool)
    assigned_parent_ids = []
    total_initial_capacity = sum(remaining_capacity.get(pid, 0) for pid in parent_ids)

    _LOG.info(
        f"[NonContext Matching] FK matching with capacity enforcement | "
        f"n_children={n_tgt} | "
        f"n_parents={len(parent_ids)} | "
        f"total_capacity={total_initial_capacity} | "
        f"capacity_deficit={n_tgt - total_initial_capacity}"
    )

    for child_idx in range(n_tgt):
        # Get row probabilities for this child
        row_probs = prob_matrix[child_idx].clone()

        # Apply penalty to parents at/over quota (instead of zeroing them out)
        row_probs[~torch.tensor(available_mask)] *= QUOTA_PENALTY_FACTOR

        # Renormalize
        row_sum = row_probs.sum()
        if row_sum > 0:
            row_probs = row_probs / row_sum

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


def initialize_remaining_capacity(
    *,
    fk_model_workspace_dir: Path,
    parent_table: DataTable,
    parent_pk: str,
) -> dict:
    """
    Initialize remaining_capacity dict using engine-based cardinality predictions.

    This function uses the trained mostlyai-engine model to predict the number of children
    each parent should have. The synthetic parent data is processed in chunks to avoid
    memory issues with large parent tables.

    Args:
        fk_model_workspace_dir: Directory containing trained models
        parent_table: Synthetic parent table
        parent_pk: Primary key column name

    Returns:
        Dictionary {parent_id: predicted_capacity}
    """
    t0 = time.time()
    cardinality_workspace_dir = fk_model_workspace_dir / "cardinality_model"
    assert cardinality_workspace_dir.exists()

    remaining_capacity = {}
    total_parents = 0

    _LOG.info(f"[NonContext Cardinality] Generate predictions | table={parent_table.name}")
    for chunk_idx, parent_chunk in enumerate(parent_table.read_chunks(do_coerce_dtypes=True)):
        chunk_size = len(parent_chunk)
        _LOG.info(f"[NonContext Cardinality] Process chunk | chunk_idx={chunk_idx} | chunk_size={chunk_size}")

        engine.generate(
            seed_data=parent_chunk[
                [col for col in parent_chunk.columns if col not in [parent_pk, CHILDREN_COUNT_COLUMN_NAME]]
            ],
            workspace_dir=cardinality_workspace_dir,
            update_progress=lambda **kwargs: None,
        )

        predicted_data_path = cardinality_workspace_dir / "SyntheticData"
        predicted_data = pd.read_parquet(predicted_data_path)
        predicted_counts = predicted_data[CHILDREN_COUNT_COLUMN_NAME].astype(int)

        parent_ids = parent_chunk[parent_pk]
        for parent_id, count in zip(parent_ids, predicted_counts):
            remaining_capacity[parent_id] = count

        total_parents += chunk_size
        shutil.rmtree(predicted_data_path)

    total_capacity = sum(remaining_capacity.values())
    _LOG.info(
        f"[NonContext Cardinality] Initialize remaining capacity | total_parents={total_parents} | total_capacity={total_capacity} | time={time.time() - t0:.2f}s"
    )

    return remaining_capacity


def match_non_context(
    *,
    fk_models_workspace_dir: Path,
    tgt_data: pd.DataFrame,
    parent_data: pd.DataFrame,
    tgt_parent_key: str,
    parent_primary_key: str,
    parent_table_name: str,
    remaining_capacity: dict,
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
        remaining_capacity: dict {parent_id: remaining_slots} for quota enforcement.
                           Enables dynamic quota enforcement - capacity is decremented
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
            f"[NonContext Matching] FK matching data | total_rows={len(tgt_data)} | null_rows={null_mask.sum()} | non_null_rows={non_null_mask.sum()}"
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
        _LOG.info(
            f"[NonContext Matching] FK matching data | total_rows={len(tgt_data)} | null_rows=0 | non_null_rows={len(tgt_data)}"
        )
        tgt_data_non_null = tgt_data.copy()
        non_null_indices = tgt_data.index.tolist()
        non_null_mask = pd.Series(True, index=tgt_data.index)

    fk_model_workspace_dir = fk_models_workspace_dir / safe_name(tgt_parent_key)
    matching_model_dir = fk_model_workspace_dir / "fk_matching_model"
    tgt_stats_dir = matching_model_dir / "tgt-stats"
    parent_stats_dir = matching_model_dir / "parent-stats"

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
        f"[NonContext Matching] FK model matching | temperature={temperature} | top_k={top_k} | top_p={top_p} | parent_sample_size={fk_parent_sample_size}"
    )

    prob_matrix = build_parent_child_probabilities(
        model=model,
        tgt_encoded=tgt_encoded,
        parent_encoded=parent_encoded,
    )

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
    _LOG.info(f"[NonContext Matching] FK matching completed | matched={n_matched} | null={n_null}")

    return tgt_data
