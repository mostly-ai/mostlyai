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

import abc
import functools
import re
from typing import Any
from urllib.parse import urlparse

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.file.base import FileContainer


class BucketBasedContainer(FileContainer, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = None
        self.bucket_path = None
        self._client = None  # assumed to be provided by the subclass

    @property
    def inner_path(self):
        return f"{self.bucket_name}/{self.bucket_path}"

    def __hash__(self):
        return hash(self.inner_path)

    @classmethod
    @abc.abstractmethod
    def cloud_path_cls(cls): ...

    @property
    @abc.abstractmethod
    def file_system(self) -> Any: ...

    @property
    @functools.cache
    def path(self):
        return self.cloud_path_cls()(cloud_path=f"{self.path_prefix}{self.inner_path}", client=self._client)

    def is_accessible(self) -> bool:
        return self._check_authenticity() and (self._is_bucket_accessible() if self.bucket_name else True)

    def _is_bucket_accessible(self):
        bucket_path = self.bucket_name if self.bucket_path is None else self.path_without_scheme
        return self.file_system.ls(bucket_path)

    @abc.abstractmethod
    def _check_authenticity(self) -> bool:
        pass

    def set_uri(self, uri: str):
        def escape_trail_slash(s: str):
            return re.sub(r"^/+", "", s)

        try:
            match = self._re_match_uri(uri)
            self.bucket_name = escape_trail_slash(match.group(1))
            self.bucket_path = escape_trail_slash(match.group(2))
        except Exception:
            raise MostlyDataException("The location must contain the full path which includes the bucket.")

    @staticmethod
    def normalize_bucket_location(location: str) -> str:
        uri = urlparse(location)
        parts = [uri.netloc.strip("/"), uri.path.strip("/")]
        joined = "/".join([part for part in parts if part])
        if "/" not in joined:
            joined = joined + "/"
        return joined

    def set_location(self, location: str) -> dict:
        location = self.normalize_bucket_location(location)
        return super().set_location(location)

    def list_locations(self, prefix: str | None) -> list[str]:
        """
        List the available locations of a given prefix.
        If the prefix is None or an empty string, it will list the buckets.
        If the prefix refers to a bucket or a directory, it will return itself and the objects under it.
        If the prefix refers to a file, it will return a list containing itself.
        If the prefix refers to a non-existent object, it will return an empty list.

        The workaround implemented here is meant to handle the inconsistent behaviors of
        the different file system and to unify the output format.

        TODO: [Known issue] GCSFileSystem can only list a folder properly when the prefix ends with a slash
        """
        locations = []
        try:
            protocol = self.path_prefix
            prefix = prefix or ""
            # strip ending slashes that are not part of the protocol
            prefix = protocol + prefix.removeprefix(protocol).rstrip("/")
            # NOTE: with gcsfs >= 2025.5.0, GCSFileSystem's exists() and isdir() do not work properly for root prefix
            # so we need to ensure that those functions are not called in this case
            is_root = prefix == protocol
            if is_root or self.file_system.exists(prefix):
                if not is_root:
                    # only add ending slash for directories
                    prefix = prefix + "/" if self.file_system.isdir(prefix) else prefix
                # Directories come first
                locations = sorted(
                    self.file_system.ls(prefix, detail=True),
                    key=lambda loc: loc["type"],
                )
                # Append an ending slash for directories if it is not there yet
                locations = [
                    loc["name"] + "/" if loc["type"] == "directory" and not loc["name"].endswith("/") else loc["name"]
                    for loc in locations
                ]
                # Add the protocol prefix to the locations if not there yet
                locations = [protocol + loc if not loc.lower().startswith(protocol) else loc for loc in locations]
                # Append the prefix itself in the list if it is not there yet
                if prefix not in locations and prefix != "/":
                    locations = [prefix] + locations
        except FileNotFoundError:
            pass
        return locations
