from mostlyai.sdk._local.mcp_server import _doc_string
from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import SyntheticProbeConfig, SyntheticTableConfig, SyntheticTableConfiguration
from typing import get_args, get_origin, Union
import inspect
from pydantic import BaseModel

def debug_collect_all_models(typ, seen_types=None):
    if seen_types is None:
        seen_types = set()
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is Union:
        for arg in args:
            yield from debug_collect_all_models(arg, seen_types)
    elif origin in (list, tuple, set, frozenset):
        if args:
            yield from debug_collect_all_models(args[0], seen_types)
    elif origin is dict:
        if len(args) == 2:
            yield from debug_collect_all_models(args[1], seen_types)
    elif inspect.isclass(typ) and issubclass(type(typ), type) and issubclass(typ, BaseModel):
        if typ not in seen_types:
            seen_types.add(typ)
            yield typ
            for field in getattr(typ, 'model_fields', {}).values():
                yield from debug_collect_all_models(field.annotation, seen_types)

def test_doc_string_function():
    # print("== MostlyAI.probe ==")
    # print(_doc_string(MostlyAI.probe))
    # print("\n== SyntheticProbeConfig (no cascade) ==")
    # print(_doc_string(SyntheticProbeConfig, cascade=False))
    print("\n== SyntheticProbeConfig (cascade) ==")
    doc = _doc_string(SyntheticProbeConfig, cascade=True)
    print(doc)
    print("\nAll model types found:")
    for t in debug_collect_all_models(SyntheticProbeConfig, set()):
        print(t.__name__)
    print("\nField annotations in SyntheticProbeConfig:")
    for fname, field in SyntheticProbeConfig.model_fields.items():
        print(f"{fname}: {field.annotation}")
    assert "SyntheticProbeConfig" in doc
    assert "SyntheticTableConfig" in doc, "Should include SyntheticTableConfig docstring in cascade"
    assert "SyntheticTableConfiguration" in doc, "Should include SyntheticTableConfiguration docstring in cascade"

test_doc_string_function()
