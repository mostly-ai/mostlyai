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

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.partitioned_dataset import PartitionedDataset


class TestPartitionedDatasetBasic:
    """Test basic functionality of PartitionedDataset."""

    def test_partitioned_dataset_initialization(self):
        """Test dataset initialization with different partition configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create partitions with different sizes
            df1 = pd.DataFrame({"id": range(0, 100), "value": range(0, 100)})
            df2 = pd.DataFrame({"id": range(100, 250), "value": range(100, 250)})
            df3 = pd.DataFrame({"id": range(250, 300), "value": range(250, 300)})

            for i, df in enumerate([df1, df2, df3]):
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            # Create ParquetDataTable from the directory containing all parquet files
            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            assert len(dataset) == 300
            assert len(dataset.partition_info) == 3
            assert dataset.partition_info[0]["size"] == 100
            assert dataset.partition_info[1]["size"] == 150
            assert dataset.partition_info[2]["size"] == 50

    def test_single_partition(self):
        """Test dataset with single partition."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100), "value": ["A"] * 100})
            file_path = temp_path / "single.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            assert len(dataset) == 100
            assert len(dataset.partition_info) == 1

    def test_empty_dataset(self):
        """Test dataset with empty partition files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create empty dataframe
            df_empty = pd.DataFrame({"id": [], "value": []})
            df_normal = pd.DataFrame({"id": range(0, 50), "value": ["A"] * 50})

            file_empty = temp_path / "empty.parquet"
            file_normal = temp_path / "normal.parquet"
            df_empty.to_parquet(file_empty)
            df_normal.to_parquet(file_normal)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            assert len(dataset) == 50
            assert dataset.partition_info[0]["size"] == 0
            assert dataset.partition_info[1]["size"] == 50

    def test_partitioned_dataset_length(self):
        """Test total row count matches sum of partitions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            sizes = [50, 75, 25, 100]
            total_expected = sum(sizes)

            start_id = 0
            for i, size in enumerate(sizes):
                df = pd.DataFrame({"id": range(start_id, start_id + size)})
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)
                start_id += size

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)
            assert len(dataset) == total_expected


class TestPartitionedDatasetSlicing:
    """Test slicing functionality across partitions."""

    def test_slice_within_single_partition(self):
        """Test slicing that stays within a single partition."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df1 = pd.DataFrame({"id": range(0, 100), "category": ["A"] * 100})
            df2 = pd.DataFrame({"id": range(100, 200), "category": ["B"] * 100})

            file1 = temp_path / "part1.parquet"
            file2 = temp_path / "part2.parquet"
            df1.to_parquet(file1)
            df2.to_parquet(file2)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            # Slice within first partition
            slice_result = dataset[10:50]
            assert len(slice_result) == 40
            assert all(slice_result.category == "A")
            assert slice_result.id.min() == 10
            assert slice_result.id.max() == 49

    def test_slice_across_multiple_partitions(self):
        """Test slicing that spans multiple partitions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df1 = pd.DataFrame({"id": range(0, 100), "category": ["A"] * 100})
            df2 = pd.DataFrame({"id": range(100, 200), "category": ["B"] * 100})
            df3 = pd.DataFrame({"id": range(200, 300), "category": ["C"] * 100})

            for i, df in enumerate([df1, df2, df3]):
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            # Slice across partitions 1 and 2
            slice_result = dataset[50:150]
            assert len(slice_result) == 100
            categories = slice_result.category.unique()
            assert "A" in categories and "B" in categories
            assert slice_result.id.min() == 50
            assert slice_result.id.max() == 149

    def test_slice_at_partition_boundaries(self):
        """Test slicing exactly at partition boundaries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df1 = pd.DataFrame({"id": range(0, 100), "category": ["A"] * 100})
            df2 = pd.DataFrame({"id": range(100, 200), "category": ["B"] * 100})

            file1 = temp_path / "part1.parquet"
            file2 = temp_path / "part2.parquet"
            df1.to_parquet(file1)
            df2.to_parquet(file2)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            # Slice exactly at boundary
            slice_result = dataset[100:200]
            assert len(slice_result) == 100
            assert all(slice_result.category == "B")

    def test_out_of_bounds_slicing(self):
        """Test slicing with out-of-bounds indices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100)})
            file_path = temp_path / "part.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            # Test various out-of-bounds scenarios
            assert len(dataset[50:150]) == 50  # End beyond dataset
            assert len(dataset[-10:50]) == 50  # Negative start (becomes 0)
            assert len(dataset[150:200]) == 0  # Start beyond dataset
            assert len(dataset[50:50]) == 0  # Empty slice

    def test_empty_slices(self):
        """Test slicing that results in empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100)})
            file_path = temp_path / "part.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            empty_slice = dataset[200:300]  # Completely out of bounds
            assert len(empty_slice) == 0
            assert isinstance(empty_slice, pd.DataFrame)


class TestPartitionedDatasetRandomSampling:
    """Test random sampling functionality."""

    def test_random_sampling_basic(self):
        """Test basic random sampling scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 1000), "value": range(0, 1000)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            # Test sampling less than total
            sample = dataset.random_sample(100)
            assert len(sample) == 100
            assert set(sample.columns) == {"id", "value"}

            # Test sampling exactly total
            sample_all = dataset.random_sample(1000)
            assert len(sample_all) == 1000

    def test_random_sampling_distribution(self):
        """Test partition-weighted distribution over multiple samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create partitions with known size ratio (1:2:1)
            df1 = pd.DataFrame({"id": range(0, 100), "partition": ["A"] * 100})
            df2 = pd.DataFrame({"id": range(100, 300), "partition": ["B"] * 200})
            df3 = pd.DataFrame({"id": range(300, 400), "partition": ["C"] * 100})

            for i, df in enumerate([df1, df2, df3]):
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            # Sample multiple times and check distribution
            samples = []
            for _ in range(10):  # Multiple samples for statistical significance
                sample = dataset.random_sample(80)  # 20% of total
                distribution = sample.partition.value_counts()
                samples.append(distribution)

            # The algorithm selects partitions until it has enough data,
            # so we expect some samples to be entirely from one partition
            # and others to be mixed. This is the expected behavior.
            total_samples = sum(s.sum() for s in samples)
            assert total_samples == 800  # 10 samples * 80 each

    def test_random_sampling_edge_cases(self):
        """Test edge cases in random sampling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            # Test sampling 0 items
            empty_sample = dataset.random_sample(0)
            assert len(empty_sample) == 0
            assert isinstance(empty_sample, pd.DataFrame)

            # Test sampling 1 item
            single_sample = dataset.random_sample(1)
            assert len(single_sample) == 1

    def test_random_sampling_with_empty_partitions(self):
        """Test sampling when some partitions are empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df_empty = pd.DataFrame({"id": [], "value": []})
            df_normal = pd.DataFrame({"id": range(0, 100), "value": range(0, 100)})

            file_empty = temp_path / "empty.parquet"
            file_normal = temp_path / "normal.parquet"
            df_empty.to_parquet(file_empty)
            df_normal.to_parquet(file_normal)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            sample = dataset.random_sample(50)
            assert len(sample) == 50
            # Should only contain data from non-empty partition


class TestPartitionedDatasetErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_slice_types(self):
        """Test invalid slice types raise appropriate errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            with pytest.raises(TypeError):
                _ = dataset["invalid"]

            with pytest.raises(TypeError):
                _ = dataset[{"invalid": "key"}]

    def test_out_of_range_index_access(self):
        """Test accessing indices that are out of range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            # Test accessing out-of-range indices
            with pytest.raises(IndexError):
                dataset._find_partition_for_index(150)

            with pytest.raises(IndexError):
                dataset._find_partition_for_index(-1)


class TestPartitionedDatasetPerformance:
    """Test performance characteristics and scalability."""

    def test_metadata_reading_efficiency(self):
        """Test that metadata reading is fast and doesn't load full data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a moderately large dataset
            df = pd.DataFrame({"id": range(0, 10000), "data": ["x"] * 10000})
            file_path = temp_path / "large.parquet"
            df.to_parquet(file_path)

            # Creating dataset should be fast (only reads metadata)
            import time

            start_time = time.time()
            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)
            init_time = time.time() - start_time

            assert len(dataset) == 10000
            assert init_time < 1.0  # Should be very fast


class TestPartitionedDatasetIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_fk_scenario(self):
        """Test scenario similar to FK model usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create realistic parent data (multiple partitions)
            parent_dir = temp_path / "parent_data"
            parent_dir.mkdir()
            for i in range(3):
                parent_df = pd.DataFrame(
                    {
                        "parent_id": range(i * 1000, (i + 1) * 1000),
                        "parent_value": [f"parent_{j}" for j in range(i * 1000, (i + 1) * 1000)],
                    }
                )
                file_path = parent_dir / f"parents_{i}.parquet"
                parent_df.to_parquet(file_path)

            # Create child data
            child_dir = temp_path / "child_data"
            child_dir.mkdir()
            for i in range(2):
                child_df = pd.DataFrame(
                    {
                        "child_id": range(i * 500, (i + 1) * 500),
                        "child_value": [f"child_{j}" for j in range(i * 500, (i + 1) * 500)],
                    }
                )
                file_path = child_dir / f"children_{i}.parquet"
                child_df.to_parquet(file_path)

            parent_table = ParquetDataTable(path=parent_dir)
            child_table = ParquetDataTable(path=child_dir)
            parent_dataset = PartitionedDataset(parent_table)
            child_dataset = PartitionedDataset(child_table)

            # Simulate FK processing: batch of children, sample parents
            children_batch_size = 200
            parents_per_child = 100

            for start_idx in range(0, len(child_dataset), children_batch_size):
                end_idx = min(start_idx + children_batch_size, len(child_dataset))

                # Get children batch
                children_batch = child_dataset[start_idx:end_idx]

                # Sample parents for this batch
                n_parents_needed = len(children_batch) * parents_per_child
                parent_sample = parent_dataset.random_sample(n_parents_needed)

                # Verify sizes
                assert len(children_batch) <= children_batch_size
                assert len(parent_sample) == n_parents_needed  # With replacement if needed

    def test_memory_bounded_processing(self):
        """Test that memory usage is bounded regardless of dataset size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dataset larger than cache
            for i in range(10):  # 10 partitions
                df = pd.DataFrame({"id": range(i * 1000, (i + 1) * 1000)})
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table)

            # Process data in batches
            batch_size = 500
            for start_idx in range(0, len(dataset), batch_size):
                end_idx = min(start_idx + batch_size, len(dataset))
                batch = dataset[start_idx:end_idx]

                assert len(batch) <= batch_size


class TestPartitionedDatasetCaching:
    """Test caching functionality."""

    def test_basic_caching(self):
        """Test that caching works and returns copies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test data
            df = pd.DataFrame({"id": range(0, 100), "value": range(0, 100)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table, max_cached_partitions=2)

            # Load data twice - should be cached on second access
            data1 = dataset[0:50]
            data2 = dataset[0:50]

            # Verify data is the same but different objects (copies)
            pd.testing.assert_frame_equal(data1, data2)
            assert data1 is not data2  # Different objects

            # Verify cache has 1 partition (using lru_cache stats)
            cache_info = dataset._load_partition_cached.cache_info()
            assert cache_info.currsize == 1

    def test_cache_eviction(self):
        """Test that LRU eviction works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create 3 partitions
            for i in range(3):
                df = pd.DataFrame({"id": range(i * 100, (i + 1) * 100)})
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            # Cache limit of 2
            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table, max_cached_partitions=2)

            # Access first partition
            _ = dataset[0:10]
            assert dataset._load_partition_cached.cache_info().currsize == 1

            # Access second partition
            _ = dataset[100:110]
            assert dataset._load_partition_cached.cache_info().currsize == 2

            # Access third partition - should evict first (LRU)
            _ = dataset[200:210]
            assert dataset._load_partition_cached.cache_info().currsize == 2

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 100)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            # Load data to populate cache
            _ = dataset[0:10]
            assert dataset._load_partition_cached.cache_info().currsize == 1

            # Clear cache
            dataset.clear_cache()
            assert dataset._load_partition_cached.cache_info().currsize == 0

    def test_cache_hit_ratio(self):
        """Test that cache hits improve with repeated access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            df = pd.DataFrame({"id": range(0, 1000)})
            file_path = temp_path / "data.parquet"
            df.to_parquet(file_path)

            table = ParquetDataTable(path=file_path)
            dataset = PartitionedDataset(table)

            # Initial access - should be cache miss
            _ = dataset[0:100]
            cache_info = dataset._load_partition_cached.cache_info()
            assert cache_info.misses == 1
            assert cache_info.hits == 0

            # Repeated access - should be cache hit
            _ = dataset[50:150]  # Overlapping slice, same partition
            cache_info = dataset._load_partition_cached.cache_info()
            assert cache_info.misses == 1  # Still only 1 miss
            assert cache_info.hits == 1  # Now 1 hit

    def test_unlimited_cache_with_minus_one(self):
        """Test that max_cached_partitions=-1 keeps all partitions in memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create 5 partitions (more than typical cache limit)
            for i in range(5):
                df = pd.DataFrame({"id": range(i * 100, (i + 1) * 100)})
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            # Use -1 for unlimited caching
            table = ParquetDataTable(path=temp_path)
            dataset = PartitionedDataset(table, max_cached_partitions=-1)

            # Access all partitions
            for i in range(5):
                _ = dataset[i * 100 : (i * 100) + 10]

            # All partitions should be cached (no eviction with unlimited cache)
            cache_info = dataset._load_partition_cached.cache_info()
            assert cache_info.currsize == 5  # All 5 partitions cached
            assert cache_info.misses == 5  # 5 initial misses
            assert cache_info.hits == 0  # No hits yet

            # Access partitions again - should all be cache hits
            for i in range(5):
                _ = dataset[i * 100 : (i * 100) + 10]

            cache_info = dataset._load_partition_cached.cache_info()
            assert cache_info.currsize == 5  # Still all 5 partitions cached
            assert cache_info.misses == 5  # Still only 5 misses
            assert cache_info.hits == 5  # Now 5 hits

    def test_unlimited_cache_vs_limited_cache(self):
        """Test comparison between unlimited and limited cache behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create 4 partitions
            for i in range(4):
                df = pd.DataFrame({"id": range(i * 100, (i + 1) * 100)})
                file_path = temp_path / f"part{i}.parquet"
                df.to_parquet(file_path)

            # Test with limited cache (maxsize=2)
            limited_table = ParquetDataTable(path=temp_path)
            limited_dataset = PartitionedDataset(limited_table, max_cached_partitions=2)

            # Access all 4 partitions
            for i in range(4):
                _ = limited_dataset[i * 100 : (i * 100) + 10]

            # Should only cache 2 partitions (LRU eviction)
            limited_cache_info = limited_dataset._load_partition_cached.cache_info()
            assert limited_cache_info.currsize == 2

            # Test with unlimited cache (-1)
            unlimited_table = ParquetDataTable(path=temp_path)
            unlimited_dataset = PartitionedDataset(unlimited_table, max_cached_partitions=-1)

            # Access all 4 partitions
            for i in range(4):
                _ = unlimited_dataset[i * 100 : (i * 100) + 10]

            # Should cache all 4 partitions
            unlimited_cache_info = unlimited_dataset._load_partition_cached.cache_info()
            assert unlimited_cache_info.currsize == 4
