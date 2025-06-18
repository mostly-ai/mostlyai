# mcp_server.py
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

import os
import pandas as pd
from pydantic import BaseModel
from fastmcp import FastMCP
from typing import Annotated, get_args, get_origin, Union
from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import Generator, Connector, SyntheticDataset
import inspect
import types
import requests

def fetch_docs(url: str, error_prefix: str) -> str:
    """
    helper function to fetch documentation from a url
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"{error_prefix}: {str(e)}"

mcp = FastMCP(
    name="MOSTLY AI MCP",
    instructions="This MCP server provides access to the Mostly AI API using the SDK. Use llms_docs for quick reference and llms_full_docs for comprehensive details."
)

@mcp.resource(
    name="llms_docs",
    description="Quick reference documentation for the MOSTLY AI SDK",
    uri="https://mostly-ai.github.io/mostlyai/llms.txt"
)
def get_llms_docs() -> str:
    return fetch_docs(
        "https://mostly-ai.github.io/mostlyai/llms.txt",
        "Error fetching quick reference docs"
    )

@mcp.resource(
    name="llms_full_docs",
    description="Comprehensive documentation and examples for the MOSTLY AI SDK",
    uri="https://mostly-ai.github.io/mostlyai/llms-full.txt"
)
def get_llms_full_docs() -> str:
    return fetch_docs(
        "https://mostly-ai.github.io/mostlyai/llms-full.txt",
        "Error fetching full documentation"
    )

def get_mostly_client():
    api_key = os.getenv("MOSTLY_API_KEY")
    if api_key:
        return MostlyAI(api_key=api_key)
    try:
        return MostlyAI(local=True)
    except Exception:
        return MostlyAI(local=True)

mostly = get_mostly_client()

def _doc_string(obj, cascade: bool = True):
    """
    return the docstring of the object. if cascade=True and the object is a pydantic BaseModel,
    recursively include docstrings for all non-primitive pydantic fields, including those in containers.
    """
    def is_primitive(typ):
        return typ in {str, int, float, bool, bytes, type(None)}

    def collect_all_models(typ, seen_types):
        origin = get_origin(typ)
        args = get_args(typ)
        if origin is Union or type(typ) is types.UnionType:
            for arg in args:
                yield from collect_all_models(arg, seen_types)
            return
        if origin in (list, tuple, set, frozenset):
            if args:
                yield from collect_all_models(args[0], seen_types)
            return
        if origin is dict:
            if len(args) == 2:
                yield from collect_all_models(args[1], seen_types)
            return
        if inspect.isclass(typ) and issubclass(type(typ), type) and issubclass(typ, BaseModel) and not is_primitive(typ):
            if typ not in seen_types:
                seen_types.add(typ)
                yield typ
                for field in getattr(typ, 'model_fields', {}).values():
                    yield from collect_all_models(field.annotation, seen_types)
            return

    def get_doc(cls, seen=None):
        if seen is None:
            seen = set()
        if cls in seen:
            return ''
        seen.add(cls)
        doc = f"{cls.__name__}:\n{inspect.getdoc(cls) or ''}\n"
        if cascade and issubclass(cls, BaseModel):
            all_models = set(collect_all_models(cls, set())) - {cls}
            for model_type in all_models:
                if model_type not in seen:
                    doc += get_doc(model_type, seen)
        return doc

    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return inspect.getdoc(obj) or ''
    elif inspect.isclass(obj):
        return get_doc(obj)
    else:
        return inspect.getdoc(obj.__class__) or ''

@mcp.tool(
    description="Train a synthetic data generator from a CSV/Parquet file or a config dict."
)
def train_generator(
    name: str,
    data: str = None,
    config: dict = None,
) -> dict[str, str]:
    g: Generator = mostly.train(name=name, data=data, config=config, start=True, wait=True)
    return dict(generator_id=g.id, name=g.name)

@mcp.tool(
    description="Create a connector and optionally validate the connection before saving."
)
def connect(config: dict, test_connection: bool = True):
    return mostly.connect(config=config, test_connection=test_connection)

@mcp.tool(
    description="Generate synthetic data from a trained generator. For full details, use introspect('generate')."
)
def generate(
    generator: str,
    config: dict = None,
    size: int = None,
    seed: dict = None,
    name: str = None,
):
    return mostly.generate(
        generator=generator,
        config=config,
        size=size,
        seed=seed,
        name=name,
    )

@mcp.tool(
    description="Probe a trained generator to get synthetic samples. For full details, use introspect('probe')."
)
def probe(
    generator_id: str,
    sample_size: int = 1,
    config: dict = None,
) -> list[dict]:
    df = mostly.probe(generator=generator_id, size=sample_size, config=config)
    return df.to_dict(orient="records") if hasattr(df, 'to_dict') else []

@mcp.tool(description=_doc_string(getattr(mostly.connectors, "get")))
def get_connector(connector_id: str):
    return mostly.connectors.get(connector_id)

@mcp.tool(description=_doc_string(getattr(mostly.connectors, "list")))
def list_connectors(
    offset: int = 0,
    limit: int = None,
    search_term: str = None,
    access_type: str = None,
    owner_id: str = None,
    visibility: str = None,
    created_from: str = None,
    created_to: str = None,
    sort_by: str = None,
):
    return list(mostly.connectors.list(
        offset=offset,
        limit=limit,
        search_term=search_term,
        access_type=access_type,
        owner_id=owner_id,
        visibility=visibility,
        created_from=created_from,
        created_to=created_to,
        sort_by=sort_by,
    ))

@mcp.tool(description=_doc_string(getattr(Connector, "delete")))
def delete_connector(connector_id: str):
    return mostly.connectors.get(connector_id).delete()

@mcp.tool(description=_doc_string(getattr(mostly.generators, "get")))
def get_generator(generator_id: str):
    return mostly.generators.get(generator_id)

@mcp.tool(description=_doc_string(getattr(mostly.generators, "list")))
def list_generators(
    offset: int = 0,
    limit: int = None,
    status: str = None,
    search_term: str = None,
    owner_id: str = None,
    visibility: str = None,
    created_from: str = None,
    created_to: str = None,
    sort_by: str = None,
):
    return list(mostly.generators.list(
        offset=offset,
        limit=limit,
        status=status,
        search_term=search_term,
        owner_id=owner_id,
        visibility=visibility,
        created_from=created_from,
        created_to=created_to,
        sort_by=sort_by,
    ))

@mcp.tool(description=_doc_string(getattr(Generator, "clone")))
def clone_generator(generator_id: str, training_status: str = "new"):
    return mostly.generators.get(generator_id).clone(training_status=training_status)

@mcp.tool(description=_doc_string(getattr(Generator, "delete")))
def delete_generator(generator_id: str):
    return mostly.generators.get(generator_id).delete()

@mcp.tool(description=_doc_string(getattr(mostly.synthetic_datasets, "get")))
def get_synthetic_dataset(synthetic_dataset_id: str):
    return mostly.synthetic_datasets.get(synthetic_dataset_id)

@mcp.tool(description=_doc_string(getattr(mostly.synthetic_datasets, "list")))
def list_synthetic_datasets(
    offset: int = 0,
    limit: int = None,
    status: str = None,
    search_term: str = None,
    owner_id: str = None,
    visibility: str = None,
    created_from: str = None,
    created_to: str = None,
    sort_by: str = None,
):
    return list(mostly.synthetic_datasets.list(
        offset=offset,
        limit=limit,
        status=status,
        search_term=search_term,
        owner_id=owner_id,
        visibility=visibility,
        created_from=created_from,
        created_to=created_to,
        sort_by=sort_by,
    ))

@mcp.tool(description=_doc_string(getattr(SyntheticDataset, "delete")))
def delete_synthetic_dataset(synthetic_dataset_id: str):
    return mostly.synthetic_datasets.get(synthetic_dataset_id).delete()

def main():
    port = int(os.getenv("MCP_PORT", "8081"))
    mcp.run(transport="sse", host="127.0.0.1", port=port, log_level="error")

if __name__ == "__main__":
    main()