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

import ast
import re

# Constant for the file path
FILE_PATH = "mostlyai/sdk/domain.py"
MODEL_FILE_PATH = "tools/model.py"

# Dictionary for enum replacements
enum_replace_dict = {
    "        'AUTO'": "        ModelEncodingType.auto",
    "        'NEW'": "        GeneratorCloneTrainingStatus.new",
    "        'CONSTANT'": "        RareCategoryReplacementMethod.constant",
}


def get_private_classes(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())

    private_classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("_"):
            private_classes.append(ast.unparse(node))
    return private_classes


def postprocess_model_file(file_path):
    # Read the contents of the file
    with open(file_path) as file:
        lines = file.readlines()

    # Modify the contents
    new_lines = []
    import_typing_updated = False
    private_classes = get_private_classes(MODEL_FILE_PATH)

    for line in lines:
        # skip filename comment and uuid import (UUID gets replaced with str)
        if "#   filename:" in line or "from uuid import" in line:
            pass
        # Add additional imports
        elif "from enum import Enum" in line:
            new_lines.append(
                "from enum import Enum\n"
                "import pandas as pd\nfrom pathlib import Path\n"
                "from pydantic import field_validator, model_validator, Discriminator\n"
                "import uuid\n"
                "import rich\n"
                "import zipfile\n"
                "import sys\n"
                "import inspect\n"
                "from mostlyai.sdk.client._base_utils import convert_to_base64, convert_to_df, read_table_from_path\n"
            )
        elif "from typing" in line and not import_typing_updated:
            # Append ', ClassVar' to the line if it doesn't already contain ClassVar
            if "ClassVar" not in line:
                line = line.rstrip() + ", ClassVar, Literal, Annotated, Union\n"
                import_typing_updated = True
            elif "Union" not in line:
                line = line.rstrip() + ", Union\n"
            new_lines.append(line)
        else:
            # Replace 'UUID' with 'str'
            new_line = line.replace("UUID", "str")

            # Apply replacements from enum_replace_dict
            for old, new in enum_replace_dict.items():
                if old in new_line:
                    new_line = new_line.replace(old, new)

            new_lines.append(new_line)

    # append private classes
    new_lines.extend(f"\n{cls}\n" for cls in private_classes)

    docstring_decorator = """
def _add_fields_to_docstring(cls):
    lines = [f"{cls.__doc__.strip()}\\n"] if cls.__doc__ else []
    lines += ["Attributes:"]
    for name, field in cls.model_fields.items():
        if name in CustomBaseModel.model_fields:
            continue
        field_str = f"  {name}"
        if field.annotation:
            if isinstance(field.annotation, type):
                annotation_str = field.annotation.__name__
            else:
                annotation_str = (
                    str(field.annotation)
                    .replace("mostlyai.sdk.domain.", "")
                    .replace("typing.", "")
                    .replace("datetime.", "")
                )
        else:
            annotation_str = ""
        field_str += f" ({annotation_str})" if field.annotation else ""
        desc_str = f" {field.description.strip()}" if field.description else ""
        examples = getattr(field, "examples", None)
        examples_str = f" Examples: {examples[0]}" if examples else ""
        if desc_str or examples_str:
            field_str += f":{desc_str}{examples_str}"
        lines.append(field_str)
    cls.__doc__ = "\\n".join(lines)
    return cls


# add fields to docstring for all the subclasses of CustomBaseModel
for _, _obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(_obj) and issubclass(_obj, CustomBaseModel) and _obj is not CustomBaseModel:
        _add_fields_to_docstring(_obj)
    """
    new_lines.append(docstring_decorator)

    # make sure secrets/ssl in Connector* classes are not included in the repr
    content = "".join(new_lines)
    content = re.sub(r"(secrets: )([^=]+?)( =)", r"\1Annotated[\2, Field(repr=False)]\3", content)
    content = re.sub(r"(ssl: )([^=]+?)( =)", r"\1Annotated[\2, Field(repr=False)]\3", content)

    # exclude icon field from serialization (svg images are too large)
    content = re.sub(
        r"(icon: str \| None = Field\(None, description=['\"]svg image['\"])\)",
        r"\1, repr=False, exclude=True)",
        content,
    )

    # Write the modified contents back to the file
    with open(file_path, "w") as file:
        file.write(content)


if __name__ == "__main__":
    # Perform postprocessing on the model file
    postprocess_model_file(FILE_PATH)
    print("Postprocessing completed.")
