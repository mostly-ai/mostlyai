# Copyright 2024-2025 MOSTLY AI
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
import enum
import logging
import os
import sys
import warnings
import webbrowser
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
)

import httpx
import rich
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from mostlyai.sdk.client._naming_conventions import (
    map_camel_to_snake_case,
    map_snake_to_camel_case,
)
from mostlyai.sdk.client.exceptions import APIError, APIStatusError

_LOG = logging.getLogger(__name__)

GET = "GET"
POST = "POST"
PATCH = "PATCH"
DELETE = "DELETE"
HttpVerb = Literal[GET, POST, PATCH, DELETE]

DEFAULT_BASE_URL = "https://app.mostly.ai"
MAX_REQUEST_SIZE = 250_000_000

T = TypeVar("T")


class _MostlyBaseClient:
    """
    Base client class, which contains all the essentials to be used by subclasses.
    """

    API_SECTION = ["api", "v2"]
    SECTION = []

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        uds: str | None = None,
        timeout: float = 60.0,
        ssl_verify: bool = True,
    ):
        self.base_url = (base_url or os.getenv("MOSTLY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("MOSTLY_API_KEY", "")
        self.local = self.api_key == "local"
        self.transport = httpx.HTTPTransport(uds=uds) if uds else None
        self.timeout = timeout
        self.ssl_verify = ssl_verify

    def headers(self):
        return {
            "Accept": "application/json",
            "X-MOSTLY-API-KEY": self.api_key,
        }

    def request(
        self,
        path: str | list[Any],
        verb: HttpVerb,
        response_type: type = dict,
        raw_response: bool = False,
        is_api_call: bool = True,
        do_json_camel_case: bool = True,
        do_response_dict_snake_case: bool = True,
        do_include_client: bool = True,
        extra_key_values: dict | None = None,
        **kwargs,
    ) -> Any:
        """
        Extended request helper method to send HTTP requests and process the response.

        Args:
            path (str | list[Any]): A single string or a list of parts of the path to concatenate.
            verb (HttpVerb): HTTP method (GET, POST, PATCH, DELETE).
            response_type (type | None): Type to cast the response into. Defaults to dict.
            raw_response (bool): Whether to return the raw response object. Defaults to False.
            is_api_call (bool): If False, skips prefixing API_SECTION and SECTION. Defaults to True.
            do_json_camel_case (bool): Convert the provided JSON to camelCase. Defaults to True.
            do_response_dict_snake_case (bool): Convert the response dictionary to snake_case. Defaults to True.
            do_include_client (bool): Include the client property in the returned object. Defaults to True.
            extra_key_values (dict | None): Additional key-value pairs to include in the response object.
            **kwargs: Additional arguments passed to the HTTP request.

        Returns:
            Processed response based on the response_type.

        Raises:
            APIStatusError: For HTTP errors (non-2XX responses).
            APIError: For network issues or request errors.
        """
        path_list = [path] if isinstance(path, str) else [str(p) for p in path]
        prefix = self.API_SECTION + self.SECTION if is_api_call else []
        full_path = [self.base_url] + prefix + path_list
        full_url = "/".join(full_path)

        kwargs["headers"] = self.headers() | kwargs.get("headers", {})

        if (request_size := _get_total_size(kwargs)) > MAX_REQUEST_SIZE:
            warnings.warn(f"The overall {request_size=} exceeds {MAX_REQUEST_SIZE}.", UserWarning)

        if "json" in kwargs and do_json_camel_case:
            if isinstance(kwargs["json"], BaseModel):
                kwargs["json"] = kwargs["json"].model_dump()
            kwargs["json"] = map_snake_to_camel_case(kwargs["json"])
        if "params" in kwargs and do_json_camel_case:
            if isinstance(kwargs["params"], BaseModel):
                kwargs["params"] = kwargs["params"].model_dump()
            kwargs["params"] = map_snake_to_camel_case(kwargs["params"])

        try:
            with httpx.Client(timeout=self.timeout, verify=self.ssl_verify, transport=self.transport) as client:
                response = client.request(method=verb, url=full_url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                json = exc.response.json()
                if "message" in json:
                    error_msg = json["message"]
                elif "detail" in json:
                    error_msg = json["detail"]
                else:
                    error_msg = "An error occurred."
            except Exception:
                error_msg = exc.response.content
            # Handle HTTP errors (not in 2XX range)
            raise APIStatusError(f"HTTP {exc.response.status_code}: {error_msg}") from None
        except httpx.ReadTimeout as exc:
            # Handle timeout errors
            raise APIError(f"Timed out while requesting {exc.request.url!r}.") from None
        except httpx.RequestError as exc:
            # Handle request errors (e.g., network issues)
            raise APIError(f"An error occurred while requesting {exc.request.url!r}.") from None

        if raw_response:
            return response

        if response.content:
            # this section could be split into a separate method
            response_json = response.json()
            if isinstance(response_json, dict) and response_type is not dict:
                if do_include_client:
                    response_json["client"] = self
                if isinstance(extra_key_values, dict):
                    response_json["extra_key_values"] = extra_key_values
            elif response_type is dict and do_response_dict_snake_case:
                response_json = map_camel_to_snake_case(response_json)
            return response_type(**response_json) if isinstance(response_json, dict) else response_json
        else:
            return None


class Paginator(Generic[T]):
    def __init__(self, client: _MostlyBaseClient, response_class: T, **kwargs):
        """
        Generic paginator for listing objects with pagination.

        Args:
            client (_MostlyBaseClient): The client instance to use for the requests.
            response_class (type[T]): Class of the objects to be listed.
            **kwargs (dict): Additional filter parameters including 'offset' and 'limit'.
        """
        self.client = client
        self.response_class = response_class
        self.offset = max(0, kwargs.pop("offset", 0))
        self.limit = kwargs.pop("limit", None)
        self.page_limit = min(self.limit, 50) if self.limit is not None else 50
        self.params = map_snake_to_camel_case(kwargs)
        self.page_items = []
        self.index_in_page = 0
        self.index = 0
        self.is_last_page = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Any cleanup if necessary
        pass

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self.limit is not None and self.index >= self.limit:
            raise StopIteration

        if self.index_in_page >= len(self.page_items):
            self._fetch_page()
            self.index_in_page = 0
            if not self.page_items:
                raise StopIteration

        item = self.page_items[self.index_in_page]
        self.index_in_page += 1
        self.index += 1
        return self.response_class(**item, client=self.client)

    def _fetch_page(self):
        if self.is_last_page:
            self.page_items = []
            return

        page_params = self.params | {"offset": self.offset, "limit": self.page_limit}
        response = self.client.request(verb=GET, path=[], params=page_params)

        self.page_items = response.get("results", [])
        total_count = response.get("total_count", 0)
        self.offset += len(self.page_items)

        if self.offset >= total_count:
            self.is_last_page = True


class CustomBaseModel(BaseModel):
    OPEN_URL_PARTS: ClassVar[list] = None  # ["d", "object-name"]
    client: Annotated[Any | None, Field(exclude=True, repr=False)] = None
    extra_key_values: Annotated[dict | None, Field(exclude=True, repr=False)] = None
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    @model_validator(mode="before")
    def __warn_extra_fields__(cls, values):
        if isinstance(values, dict):
            extra_fields = values.keys() - cls.model_fields.keys() - {v.alias for v in cls.model_fields.values()}
            if extra_fields:
                _LOG.warning(f"ignoring unrecognized fields for {cls.__name__}: {', '.join(extra_fields)}")
        return values

    def _repr_html_(self):
        # Use rich.print to create a rich representation of the model
        console = Console()
        with console.capture() as capture:
            # dump the model while making sure attributes with repr=False is excluded
            safe_model_dump = {k: v for k, v in self.__repr_args__()}
            rich.print(safe_model_dump)
        return capture.get()

    def open(self) -> None:
        """
        Opens the instance in a web browser.
        """
        if self.client is None or not self.OPEN_URL_PARTS or not hasattr(self, "id"):
            raise APIError("Cannot open the instance")
        url = "/".join([self.client.base_url, *self.OPEN_URL_PARTS, str(self.id)])
        rich.print(f"Go to [link={url}]{url}[/]")
        webbrowser.open_new(url)

    def reload(self):
        """
        Reload the instance to reflect its current state.
        """
        if hasattr(self.client, "get"):
            reloaded = self.client.get(self.id)
            for key, value in reloaded.__dict__.items():
                current_attr = getattr(self, key, None)

                # If the current attribute is a class instance, try updating it instead of overwriting
                if (
                    isinstance(current_attr, type(value))
                    and hasattr(current_attr, "__dict__")
                    and not isinstance(current_attr, enum.Enum)
                ):
                    current_attr.__dict__.update(value.__dict__)
                elif hasattr(self, key):
                    # Otherwise, directly overwrite the attribute
                    setattr(self, key, value)


def _get_total_size(obj, seen=None):
    """Recursively finds size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Mark as seen *before* entering recursion to gracefully handle self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([_get_total_size(v, seen) + _get_total_size(k, seen) for k, v in obj.items()])

    elif hasattr(obj, "__dict__"):
        size += _get_total_size(obj.__dict__, seen)

    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([_get_total_size(i, seen) for i in obj])

    return size
