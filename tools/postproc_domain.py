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

# Constant for the file path
FILE_PATH = "mostlyai/sdk/domain.py"
MODEL_FILE_PATH = "tools/model.py"

# Dictionary for enum replacements
enum_replace_dict = {
    "        'AUTO'": "        ModelEncodingType.auto",
    "        'NEW'": "        GeneratorCloneTrainingStatus.new",
    "        'CONSTANT'": "        RareCategoryReplacementMethod.constant",
    "Field('SOURCE'": "Field(ConnectorAccessType.source",
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
        # Remove filename comment
        if "#   filename:" in line:
            pass
        # Add additional imports
        elif "import UUID" in line:
            new_lines.append(
                "import pandas as pd\nfrom pathlib import Path\n"
                "from pydantic import field_validator, model_validator\n"
                "import uuid\n"
                "import rich\n"
                "import zipfile\n"
                "from mostlyai.sdk.client._base_utils import convert_to_base64\n"
            )
        elif "from typing" in line and not import_typing_updated:
            # Append ', ClassVar' to the line if it doesn't already contain ClassVar
            if "ClassVar" not in line:
                line = line.rstrip() + ", ClassVar, Literal, Annotated\n"
                import_typing_updated = True
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

    # Write the modified contents back to the file
    with open(file_path, "w") as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    # Perform postprocessing on the model file
    postprocess_model_file(FILE_PATH)
    print("Postprocessing completed.")
