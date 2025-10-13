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

from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class PartitionedDataset:
    """Common abstraction for accessing partitioned data with slicing and random sampling."""

    def __init__(self, partition_files: list[Path], max_cached_partitions: int = 3):
        self.partition_files = partition_files
        self.max_cached_partitions = max_cached_partitions
        self.partition_info = []
        self.total_rows = 0

        # Create cached method with instance-specific maxsize
        # If max_cached_partitions is -1, set maxsize to None for unlimited cache
        cache_maxsize = None if max_cached_partitions == -1 else max_cached_partitions
        self._load_partition_cached = lru_cache(maxsize=cache_maxsize)(self._load_partition_uncached)

        # Build partition index with fast metadata reads
        for file in partition_files:
            partition_size = self._get_row_count_fast(file)
            self.partition_info.append(
                {
                    "file": file,
                    "start_idx": self.total_rows,
                    "end_idx": self.total_rows + partition_size,
                    "size": partition_size,
                }
            )
            self.total_rows += partition_size

    def __getitem__(self, key) -> pd.DataFrame:
        """Support slicing: dataset[start:end]"""
        if isinstance(key, slice):
            return self._slice_data(key.start or 0, key.stop or self.total_rows)
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
        return self.total_rows

    def _get_row_count_fast(self, file_path: Path) -> int:
        """Get row count from parquet metadata without reading data."""
        try:
            parquet_file = pq.ParquetFile(file_path)
            return parquet_file.metadata.num_rows
        except Exception:
            # Fallback for non-parquet files or corrupted metadata
            df = pd.read_parquet(file_path)
            return len(df)

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
        raise IndexError(f"Index {global_idx} out of range [0, {self.total_rows})")

    def _slice_data(self, start: int, end: int) -> pd.DataFrame:
        """Load data for slice range [start:end]"""
        if start >= self.total_rows or end <= 0:
            return pd.DataFrame()

        start = max(0, start)
        end = min(self.total_rows, end)

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

    def clear_cache(self) -> None:
        """Clear the partition cache."""
        self._load_partition_cached.cache_clear()

    def iter_partitions(self) -> Iterator[tuple[int, Path, pd.DataFrame]]:
        """Iterate over partitions yielding (index, file_path, dataframe)."""
        for idx, partition in enumerate(self.partition_info):
            file_path = partition["file"]
            data = self._load_partition(file_path)
            yield idx, file_path, data

    @property
    def files(self) -> list[Path]:
        """Get the list of partition files."""
        return [partition["file"] for partition in self.partition_info]
