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

import logging
from typing import Any

import duckdb
import gcsfs
from cloudpathlib.gs import GSClient, GSPath
from google.cloud import storage

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer
from mostlyai.sdk._data.util.common import validate_gcs_key_file

_LOG = logging.getLogger(__name__)


class GcsContainer(BucketBasedContainer):
    SCHEMES = ["http", "https", "gs"]
    DEFAULT_SCHEME = "gs"
    DELIMITER_SCHEMA = "gs"
    SECRET_ATTR_NAME = "key_file"

    def __init__(self, *args, key_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_file = key_file
        self.decrypt_secret()

        if not self.key_file:
            raise MostlyDataException("Provide the key file.")
        try:
            self.client = storage.Client.from_service_account_info(self.key_file)
            self.fs = gcsfs.GCSFileSystem(project=self.key_file["project_id"], token=self.key_file)
            self._client = GSClient(project=self.key_file["project_id"], storage_client=self.client)
        except Exception as e:
            error_message = str(e).lower()
            if "unsupported algorithm" in error_message:
                raise MostlyDataException("Key file is incorrect.")

    def decrypt_secret(self, secret_attr_name: str | None = None) -> None:
        super().decrypt_secret()
        self.key_file = validate_gcs_key_file(self.key_file)
        if not self.key_file:
            raise MostlyDataException("Key file is incorrect.")

    @classmethod
    def cloud_path_cls(cls):
        return GSPath

    @property
    def storage_options(self) -> dict:
        return self.fs.storage_options

    @property
    def transport_params(self) -> dict | None:
        return dict(client=self.client)

    @property
    def file_system(self) -> Any:
        return self.fs

    def _check_authenticity(self) -> bool:
        return gcsfs.GCSFileSystem(project=self.key_file["project_id"], token=self.key_file) is not None

    def _init_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        # register the GCS filesystem with DuckDB using service account credentials
        # this is an alternative to HMAC keys, which we don't use
        con.register_filesystem(self.fs)
