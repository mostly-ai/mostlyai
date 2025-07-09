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
import re
import tempfile
from typing import Any
from urllib.parse import urlparse

import boto3
import duckdb
import s3fs
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from cloudpathlib.s3 import S3Client, S3Path

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer

_LOG = logging.getLogger(__name__)


class AwsS3FileContainer(BucketBasedContainer):
    SCHEMES = ["http", "https", "s3"]
    DEFAULT_SCHEME = "s3"
    DELIMITER_SCHEMA = "s3"
    SECRET_ATTR_NAME = "secret_key"
    ROLE_SESSION_NAME = "mostly-ai-session"

    def __init__(
        self,
        *args,
        access_key: str | None = None,
        secret_key: str | None = None,
        role_arn: str | None = None,
        external_id: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        do_decrypt_secret: bool | None = True,
        ssl_enabled: bool | None = None,
        ca_certificate: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.endpoint_url = endpoint_url or None
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = None
        self.role_arn = role_arn
        self.external_id = external_id
        # SSL
        self.ssl_enabled = ssl_enabled
        self.ca_certificate = ca_certificate
        self.no_sign_request = False
        if self.role_arn or self.external_id:
            if not os.getenv("MOSTLY_EXTERNAL_IAM_ROLE"):
                raise MostlyDataException(
                    "IAM role authentication is disabled because `MOSTLY_EXTERNAL_IAM_ROLE` is not set. Use access key and secret key instead."
                )
            if not (self.role_arn and self.external_id):
                raise MostlyDataException("Both role ARN and external ID must be provided together.")
            if do_decrypt_secret:
                self.decrypt_secret(secret_attr_name="external_id")
            self._assume_user_role()
        elif self.access_key or self.secret_key:
            if not (self.access_key and self.secret_key):
                raise MostlyDataException("Both access key and secret key must be provided together.")
            if do_decrypt_secret:
                self.decrypt_secret()  # decrypt with the default SECRET_ATTR_NAME
        else:
            _LOG.info("No credentials or IAM role provided. Using anonymous access.")
            self.access_key, self.secret_key, self.session_token = self._assume_external_role()
            self.no_sign_request = True

        self.ca_cert_content = self.decrypt(self.ca_certificate) if self.ssl_enabled and self.ca_certificate else None
        self.region = region
        # extract region from endpoint URL if it's an AWS endpoint
        if self.endpoint_url and ".amazonaws.com" in self.endpoint_url:
            match = re.search(r"s3[.-](.+?)\.amazonaws\.com", self.endpoint_url)
            if match:
                self.region = match.group(1)
        boto_session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token,
        )
        self._client = S3Client(
            boto3_session=boto_session,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token,
            no_sign_request=self.no_sign_request,
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
            if self.region:
                client_kwargs["region_name"] = self.region
            self.fs = s3fs.S3FileSystem(
                endpoint_url=self.endpoint_url,
                secret=self.secret_key,
                key=self.access_key,
                token=self.session_token,
                client_kwargs=client_kwargs,
                cache_regions=True,
                anon=self.no_sign_request,
            )
            # S3FileSystem use `anon` parameter instead of client_kwargs["config"] for anonymous access
            # so we only have to add this key value after initializing S3FileSystem
            if self.no_sign_request:
                client_kwargs["config"] = BotoConfig(signature_version=UNSIGNED)
            self._boto_resource = boto_session.resource("s3", endpoint_url=self.endpoint_url, **client_kwargs)
            self._boto_client = boto_session.client("s3", endpoint_url=self.endpoint_url, **client_kwargs)
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
                    aws_session_token=self.session_token,
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

    def _assume_external_role(self) -> tuple[str, str, str]:
        sts_client = boto3.client("sts")
        ext_role_creds = sts_client.assume_role(
            RoleArn=os.environ["MOSTLY_EXTERNAL_IAM_ROLE"],
            RoleSessionName=os.getenv("HOSTNAME", "mostlyai"),
        )["Credentials"]
        return (
            ext_role_creds["AccessKeyId"],
            ext_role_creds["SecretAccessKey"],
            ext_role_creds["SessionToken"],
        )

    def _assume_user_role(self) -> None:
        # 1. Assume Mostly's external IAM role
        ext_role_access_key, ext_role_secret_key, ext_role_session_token = self._assume_external_role()
        # 2. On behalf of the external IAM role, assume the user's role
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=ext_role_access_key,
            aws_secret_access_key=ext_role_secret_key,
            aws_session_token=ext_role_session_token,
        )
        user_role_creds = sts_client.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=self.ROLE_SESSION_NAME,
            ExternalId=self.external_id,
        )["Credentials"]
        self.access_key = user_role_creds["AccessKeyId"]
        self.secret_key = user_role_creds["SecretAccessKey"]
        self.session_token = user_role_creds["SessionToken"]

    def _init_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        # fallback to con.register_filesystem (instead of DuckDB's httpfs + aws) if:
        # 1. no endpoint is set (defaults to AWS, but region is not specified)
        # 2. it's an amazon endpoint but no region is set
        # 3. CA certificate is being used
        if (
            not self.endpoint_url
            or (self.endpoint_url and ".amazonaws.com" in self.endpoint_url and not self.region)
            or (self.ssl_enabled and self.ssl_verify)
        ):
            con.register_filesystem(self.fs)
            return

        # extract only the hostname (and optional port) from the endpoint URL
        endpoint = urlparse(self.endpoint_url).netloc
        secret_params = {
            "TYPE": "s3",
            "KEY_ID": self.access_key,
            "SECRET": self.secret_key,
            "ENDPOINT": endpoint,
            "USE_SSL": bool(self.ssl_enabled),
        }

        if self.region:
            secret_params["REGION"] = self.region

        self._create_duckdb_secret(con, secret_params)
