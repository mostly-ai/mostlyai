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

import pandas as pd
import pyarrow.dataset as ds

from mostlyai.sdk._data.file.base import FileContainer, FileDataTable, LocalFileContainer


class FeatherDataTable(FileDataTable):
    DATA_TABLE_TYPE = "feather"
    IS_WRITE_APPEND_ALLOWED = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def container_class(cls) -> type["FileContainer"]:
        return LocalFileContainer

    def _get_dataset_format(self):
        return ds.FeatherFileFormat()

    def write_data(self, df: pd.DataFrame, if_exists: str = "replace", **kwargs):
        self.handle_if_exists(if_exists)  # will gracefully handle append as replace
        df.to_feather(
            self.container.path_str,
            storage_options=self.container.storage_options,
        )
