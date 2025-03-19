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

import re
from collections.abc import Callable


def _snake_to_camel(snake_str: str) -> str:
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _camel_to_snake(camel_str: str) -> str:
    # handle acronyms by treating consecutive uppercase letters as a group
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", camel_str)  # e.g., "GPUTime" -> "GPU_Time"
    s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)  # e.g., "virtualGPU" -> "virtual_GPU"
    return s2.lower()


def _convert_case(input_data: dict, conv_func: Callable[[str], str]) -> dict:
    if not isinstance(input_data, dict):
        return input_data

    new_dict = {}
    for key, value in input_data.items():
        new_key = conv_func(key)
        # recursively convert nested dictionaries or lists
        if isinstance(value, dict):
            new_dict[new_key] = _convert_case(value, conv_func)
        elif isinstance(value, list):
            new_dict[new_key] = [_convert_case(item, conv_func) if isinstance(item, dict) else item for item in value]
        else:
            new_dict[new_key] = value
    return new_dict


def map_snake_to_camel_case(input_dict: dict) -> dict:
    return _convert_case(input_dict, _snake_to_camel)


def map_camel_to_snake_case(input_dict: dict) -> dict:
    return _convert_case(input_dict, _camel_to_snake)
