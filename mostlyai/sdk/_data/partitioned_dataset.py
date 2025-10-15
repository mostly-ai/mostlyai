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

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mostlyai.sdk._data.file.base import FileDataTable


class PartitionedDataset:
    """Cached wrapper for FileDataTable with slicing and random sampling capabilities."""

    def __init__(self, table: FileDataTable, max_cached_partitions: int = 1):
        self.table = table
        self.max_cached_partitions = max_cached_partitions
        self.partition_info = []

        # Create cached method with instance-specific maxsize
        # If max_cached_partitions is -1, set maxsize to None for unlimited cache
        cache_maxsize = None if max_cached_partitions == -1 else max_cached_partitions
        self._load_partition_cached = lru_cache(maxsize=cache_maxsize)(self._load_partition_uncached)

        # Build partition index using table information
        self._build_partition_index()

    def _build_partition_index(self):
        """Build partition index using table's files."""
        current_total = 0
        for file in self.table.files:
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

        # Randomly select partitions until we have enough rows
        selected_partitions = set()
        total_available = 0
        available_partitions = list(range(len(self.partition_info)))
        np.random.shuffle(available_partitions)

        for partition_idx in available_partitions:
            if total_available >= n_items:
                break
            selected_partitions.add(partition_idx)
            total_available += self.partition_info[partition_idx]["size"]

        # Load selected partitions
        all_data = []
        for partition_idx in selected_partitions:
            partition = self.partition_info[partition_idx]
            df = self._load_partition(partition["file"])
            all_data.append(df)

        # Combine all loaded data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sample exactly n_items from the combined data
        if len(combined_df) >= n_items:
            sampled_df = combined_df.sample(n=n_items, replace=False).reset_index(drop=True)
        else:
            # If we still don't have enough, sample with replacement
            sampled_df = combined_df.sample(n=n_items, replace=True).reset_index(drop=True)

        return sampled_df

    def __len__(self) -> int:
        return self.table.row_count

    def _get_row_count_fast(self, file_path: Path) -> int:
        """Get row count from parquet metadata without reading data."""
        # For single-partition tables, use table's row_count directly
        if len(self.table.files) == 1:
            return self.table.row_count

        # Use PyArrow parquet metadata for efficiency
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows

    def _load_partition_uncached(self, file_path: Path) -> pd.DataFrame:
        """Load partition data from disk (no caching)."""
        return pd.read_parquet(file_path)

    def _load_partition(self, file_path: Path) -> pd.DataFrame:
        """Load partition with caching."""
        return self._load_partition_cached(file_path).copy()  # Return copy to prevent mutations

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

        # Find which partitions we need
        needed_partitions = []
        for partition in self.partition_info:
            if partition["end_idx"] > start and partition["start_idx"] < end:
                # Calculate local slice within this partition
                local_start = max(0, start - partition["start_idx"])
                local_end = min(partition["size"], end - partition["start_idx"])
                needed_partitions.append((partition, local_start, local_end))

        # Load needed partitions and extract slices
        result_dfs = []
        for partition, local_start, local_end in needed_partitions:
            df = self._load_partition(partition["file"])
            slice_df = df.iloc[local_start:local_end]
            result_dfs.append(slice_df)

        return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()

    def iter_partitions(self) -> Iterator[tuple[int, Path, pd.DataFrame]]:
        """Iterate over partitions using table's method."""
        yield from self.table.iter_partitions()

    @property
    def files(self) -> list[Path]:
        """Access partition files using table's files."""
        return self.table.files

    def clear_cache(self) -> None:
        """Clear the partition cache."""
        self._load_partition_cached.cache_clear()
