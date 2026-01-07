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

import pytest
from datamodel_code_generator.reference import camel_to_snake

from mostlyai.sdk.client._naming_conventions import (
    _snake_to_camel,
    map_camel_to_snake_case,
    map_snake_to_camel_case,
)


@pytest.mark.parametrize(
    "snake_str, expected",
    [
        ("snake_case", "snakeCase"),
        ("this_is_a_test", "thisIsATest"),
        ("alreadyCamelCase", "alreadyCamelCase"),
    ],
)
def test__snake_to_camel_param(snake_str, expected):
    assert _snake_to_camel(snake_str) == expected


def test_map_snake_to_camel_case():
    input_data = {
        "snake_case_key": "value",
        "nested_dict": {
            "another_snake_case": 123,
            "deep_nested_dict": {
                "yet_another_key": "deepValue",
                "someCamelCaseKey": "someOtherValue",
                "normalNumber": 42,
            },
        },
        "list_of_dicts": [{"list_snake_case": "item1"}, {"another_list_key": "item2"}],
        "mixed_list": ["simpleString", {"nested_snake": "nestedValue"}, 3.14],
        "alreadyCamelCase": "unchangedValue",
        "no_underscore_key": "noChange",
        "empty_string_key": "",
    }

    expected_output = {
        "snakeCaseKey": "value",
        "nestedDict": {
            "anotherSnakeCase": 123,
            "deepNestedDict": {
                "yetAnotherKey": "deepValue",
                "someCamelCaseKey": "someOtherValue",
                "normalNumber": 42,
            },
        },
        "listOfDicts": [{"listSnakeCase": "item1"}, {"anotherListKey": "item2"}],
        "mixedList": ["simpleString", {"nestedSnake": "nestedValue"}, 3.14],
        "alreadyCamelCase": "unchangedValue",
        "noUnderscoreKey": "noChange",
        "emptyStringKey": "",
    }

    assert map_snake_to_camel_case(input_data) == expected_output


@pytest.mark.parametrize(
    "camel_str, expected",
    [
        ("camelCase", "camel_case"),
        ("thisIsATest", "this_is_a_test"),
        ("already_snake_case", "already_snake_case"),
        ("HTTPRequest", "http_request"),
        ("parseJSONResponse", "parse_json_response"),
        ("XMLHttpRequest", "xml_http_request"),
        ("totalVirtualGPUTime", "total_virtual_gpu_time"),
        ("totalVirtualCPUTime", "total_virtual_cpu_time"),
        ("userIDToken", "user_id_token"),
        ("AWSConfig", "aws_config"),
        ("useHTTPSConnection", "use_https_connection"),
    ],
)
def test_camel_to_snake(camel_str, expected):
    assert camel_to_snake(camel_str) == expected


def test_map_camel_to_snake_case():
    input_data = {
        "camelCaseKey": "value",
        "nestedDict": {
            "anotherCamelCase": 123,
            "deepNestedDict": {
                "yetAnotherKey": "deepValue",
                "some_snake_case_key": "someOtherValue",
                "NormalNumber": 42,
            },
        },
        "listOfDicts": [{"listCamelCase": "item1"}, {"anotherListKey": "item2"}],
        "mixedList": ["simpleString", {"nestedCamel": "nestedValue"}, 3.14],
        "AlreadySnakeCase": "unchangedValue",
        "NoUnderscoreKey": "noChange",
        "EmptyStringKey": "",
    }

    expected_output = {
        "camel_case_key": "value",
        "nested_dict": {
            "another_camel_case": 123,
            "deep_nested_dict": {
                "yet_another_key": "deepValue",
                "some_snake_case_key": "someOtherValue",
                "normal_number": 42,
            },
        },
        "list_of_dicts": [{"list_camel_case": "item1"}, {"another_list_key": "item2"}],
        "mixed_list": ["simpleString", {"nested_camel": "nestedValue"}, 3.14],
        "already_snake_case": "unchangedValue",
        "no_underscore_key": "noChange",
        "empty_string_key": "",
    }

    assert map_camel_to_snake_case(input_data) == expected_output
