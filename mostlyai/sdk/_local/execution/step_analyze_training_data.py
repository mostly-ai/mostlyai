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


from collections import ChainMap, defaultdict
from collections.abc import Callable
from pathlib import Path

from mostlyai.engine.domain import DifferentialPrivacyConfig
from mostlyai.sdk.domain import Generator, ModelEncodingType, ModelType, SourceColumnValueRange


def execute_step_analyze_training_data(
    *,
    generator: Generator,
    model_type: ModelType,
    target_table_name: str,
    workspace_dir: Path,
    update_progress: Callable,
) -> tuple[dict[str, ModelEncodingType], dict[str, SourceColumnValueRange]]:
    # import ENGINE here to avoid pre-mature loading of large ENGINE dependencies
    from mostlyai import engine
    from mostlyai.engine._workspace import Workspace

    # fetch model_config
    tgt_table = next(t for t in generator.tables if t.name == target_table_name)
    if model_type == ModelType.language:
        model_config = tgt_table.language_model_configuration
    else:
        model_config = tgt_table.tabular_model_configuration

    # convert from SDK domain to ENGINE domain
    if model_config.differential_privacy:
        differential_privacy = DifferentialPrivacyConfig(**model_config.differential_privacy.model_dump())
    else:
        differential_privacy = None

    # call ANALYZE
    engine.analyze(
        workspace_dir=workspace_dir,
        value_protection=model_config.value_protection,
        differential_privacy=differential_privacy,
        update_progress=update_progress,
    )

    # read stats
    workspace = Workspace(workspace_dir)
    tgt_stats = workspace.tgt_stats.read()
    encoding_types = _get_encoding_types(tgt_stats)
    value_ranges = _get_value_ranges(tgt_stats)
    return encoding_types, value_ranges


def _get_encoding_types(stats: dict) -> dict[str, ModelEncodingType]:
    encoding_types = {}
    for col, col_stats in stats.get("columns", {}).items():
        encoding_type = col_stats.get("encoding_type")
        if encoding_type is not None:
            encoding_types[col] = ModelEncodingType(encoding_type)
    return encoding_types


def _get_value_ranges(stats: dict) -> dict[str, SourceColumnValueRange]:
    # import ENGINE here to avoid pre-mature loading of large ENGINE dependencies
    from mostlyai.engine._encoding_types.tabular.categorical import CATEGORICAL_NULL_TOKEN, CATEGORICAL_UNKNOWN_TOKEN

    def parse_values(col_stats: dict) -> dict:
        size_limit = 1_000
        values = [
            code
            for code in col_stats.get("codes", {}).keys()
            if code not in [CATEGORICAL_UNKNOWN_TOKEN, CATEGORICAL_NULL_TOKEN]
        ][:size_limit]
        return {"values": values}

    def parse_min_max(col_stats: dict) -> dict:
        values = col_stats.get("bins", []) + col_stats.get("min5", []) + col_stats.get("max5", [])
        min_ = str(min(values)) if values else None
        max_ = str(max(values)) if values else None
        return {"min": min_, "max": max_}

    def parse_has_null(col_stats: dict) -> dict:
        has_null = any(
            [
                CATEGORICAL_NULL_TOKEN in col_stats.get("codes", {}).keys(),
                col_stats.get("has_nan", False),
                col_stats.get("has_na", False),
            ]
        )
        return {"has_null": has_null}

    def combine(*parsers):
        def pipe(col_stats: dict) -> SourceColumnValueRange:
            return SourceColumnValueRange(**ChainMap(*[parser(col_stats) for parser in parsers]))

        return pipe

    parsers = defaultdict(
        lambda: combine(parse_has_null),
        {
            ModelEncodingType.tabular_categorical: combine(parse_values, parse_has_null),
            ModelEncodingType.tabular_numeric_discrete: combine(parse_values, parse_has_null),
            ModelEncodingType.tabular_numeric_binned: combine(parse_min_max, parse_has_null),
            ModelEncodingType.tabular_numeric_digit: combine(parse_min_max, parse_has_null),
            ModelEncodingType.tabular_datetime: combine(parse_min_max, parse_has_null),
        },
    )

    value_ranges = {}
    for col, col_stats in stats.get("columns", {}).items():
        encoding_type = col_stats.get("encoding_type")
        value_ranges[col] = parsers[encoding_type](col_stats)

    return value_ranges
