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
import os
import tempfile
from typing import Any

import boto3 as boto3
import s3fs
from cloudpathlib.s3 import S3Client, S3Path

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer


_LOG = logging.getLogger(__name__)


class AwsS3FileContainer(BucketBasedContainer):
    SCHEMES = ["http", "https", "s3"]
    DEFAULT_SCHEME = "s3"
    DELIMITER_SCHEMA = "s3"
    SECRET_ATTR_NAME = "secret_key"

    def __init__(
        self,
        *args,
        access_key,
        secret_key,
        endpoint_url=None,
        do_decrypt_secret=True,
        ssl_enabled=None,
        ca_certificate=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        # SSL
        self.ssl_enabled = ssl_enabled
        self.ca_certificate = ca_certificate
        if do_decrypt_secret:
            self.decrypt_secret()
        self.ca_cert_content = self.decrypt(self.ca_certificate) if self.ssl_enabled and self.ca_certificate else None

        if not (self.secret_key and self.access_key):
            raise MostlyDataException("Provide the S3 credentials.")
        boto_session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        self._client = S3Client(
            boto3_session=boto_session,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        with tempfile.NamedTemporaryFile(delete=False) as ca_cert_file:
            if self.ca_cert_content:
                ca_cert_file.write(self.ca_cert_content.encode())
                ca_cert_file.flush()
                os.chmod(ca_cert_file.name, 0o600)
                self.ssl_verify = ca_cert_file.name
            else:
                self.ssl_verify = None
            client_kwargs = {"verify": self.ssl_verify}
            if self.endpoint_url and ".amazonaws.com" in self.endpoint_url:
                # aws s3 needs region_name to be set when using custom endpoint
                parts = self.endpoint_url.split(".")
                if len(parts) > 1:
                    client_kwargs["region_name"] = parts[1]
            self.fs = s3fs.S3FileSystem(
                endpoint_url=self.endpoint_url,
                secret=self.secret_key,
                key=self.access_key,
                client_kwargs=client_kwargs,
            )
            self._boto_resource = boto_session.resource("s3", endpoint_url=endpoint_url, verify=self.ssl_verify)
            self._boto_client = boto_session.client("s3", endpoint_url=self.endpoint_url, verify=self.ssl_verify)
        # patch CloudPath to use the same boto3 resource/client
        self._client.s3 = self._boto_resource
        self._client.client = self._boto_client

    def __del__(self):
        if self.ssl_verify and os.path.exists(self.ssl_verify):
            os.remove(self.ssl_verify)

    @classmethod
    def cloud_path_cls(cls):
        return S3Path

    @property
    def storage_options(self) -> dict:
        return self.fs.storage_options

    @property
    def transport_params(self) -> dict | None:
        return dict(client=self._boto_client)

    @property
    def file_system(self) -> Any:
        return self.fs

    def _check_authenticity(self) -> bool:
        try:
            if self.endpoint_url:
                # Check if the bucket exists
                _LOG.info(f"endpoint url: {self.endpoint_url}")
                _LOG.info(f"bucket path: {self.bucket_path}")
                _LOG.info(f"bucket name: {self.bucket_name}")
                _LOG.info(f"access key: {self.path_without_scheme}")
                _LOG.info(f"Testing ls on `{self.bucket_name}` for authenticity.")
                return self.fs.ls(self.bucket_name or "") is not None
            else:
                # Use STS to get caller identity
                # It is more reliable, in the case of a limited access (e.g. if not allowed to query bucket names)
                sts_client = boto3.client(
                    "sts",
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    endpoint_url=self.endpoint_url,
                    verify=self.ssl_verify,
                )
                sts_client.get_caller_identity()
                return True
        except Exception as e:
            error_message = str(e).lower()
            if "invalidclienttokenid" in error_message or "access key id" in error_message:
                raise MostlyDataException("Access key is incorrect.")
            elif "secret access key" in error_message or "signature" in error_message:
                raise MostlyDataException("Secret key is incorrect.")
            elif "endpoint url" in error_message:
                raise MostlyDataException("Cannot reach the endpoint URL.")
            else:
                raise MostlyDataException(f"Error has occurred: {str(e)}")
