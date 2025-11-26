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

"""constraint transformation utilities for fixed combinations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mostlyai.sdk.domain import FixedCombination


def get_constraint_meta_path(workspace_dir: Path) -> Path:
    """get constraint metadata directory path in ModelStore."""
    path = workspace_dir / "ModelStore"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ConstraintTranslator:
    """translates data between user schema and internal schema for constraints."""

    def __init__(self, constraints: list[FixedCombination]):
        """initialize translator with constraint definitions.

        Args:
            constraints: List of FixedCombination constraint objects.
        """
        self.constraints = constraints
        self.merged_columns: list[tuple[list[str], str]] = []

        # precompute merged column names
        for constraint in constraints:
            columns = constraint.columns
            merged_name = "|".join(columns)
            self.merged_columns.append((columns, merged_name))

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from user schema to internal schema.

        Args:
            df: DataFrame with original column structure.

        Returns:
            DataFrame with constrained columns merged.
        """
        df = df.copy()

        for columns, merged_name in self.merged_columns:
            # merge columns into one with | separator
            df[merged_name] = df[columns].astype(str).agg("|".join, axis=1)
            # drop original columns
            df = df.drop(columns=columns)

        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from internal schema back to user schema.

        Args:
            df: DataFrame with merged constraint columns.

        Returns:
            DataFrame with original column structure.
        """
        df = df.copy()

        for columns, merged_name in self.merged_columns:
            if merged_name in df.columns:
                # split merged column back into original columns
                split_values = df[merged_name].str.split("|", n=len(columns) - 1, expand=True)
                for i, col in enumerate(columns):
                    df[col] = split_values[i]
                # drop merged column
                df = df.drop(columns=[merged_name])

        return df

    def get_internal_columns(self, original_columns: list[str]) -> list[str]:
        """get list of column names in internal schema.

        Args:
            original_columns: List of original column names.

        Returns:
            List of column names with constraint columns merged.
        """
        columns_to_remove = set()
        columns_to_add = []

        for columns, merged_name in self.merged_columns:
            columns_to_remove.update(columns)
            columns_to_add.append(merged_name)

        internal_columns = [c for c in original_columns if c not in columns_to_remove]
        internal_columns.extend(columns_to_add)

        return internal_columns

    def to_dict(self) -> dict[str, Any]:
        """serialize translator to dictionary.

        Returns:
            Dictionary representation of the translator.
        """
        return {
            "constraints": [constraint.model_dump() for constraint in self.constraints],
            "merged_columns": [(cols, name) for cols, name in self.merged_columns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintTranslator:
        """deserialize translator from dictionary.

        Args:
            data: Dictionary representation from to_dict().

        Returns:
            ConstraintTranslator instance.
        """
        constraints = [FixedCombination(**constraint_data) for constraint_data in data["constraints"]]
        translator = cls(constraints)
        translator.merged_columns = [(cols, name) for cols, name in data["merged_columns"]]
        return translator

    def save_metadata(self, workspace_dir: Path, table_name: str, original_columns: list[str] | None = None) -> None:
        """save constraint metadata to workspace.

        Args:
            workspace_dir: Workspace directory path.
            table_name: Name of the table.
            original_columns: Original column names before transformation (optional).
        """
        meta_dir = get_constraint_meta_path(workspace_dir)
        constraints_file = meta_dir / "constraints.json"

        # load existing constraints (if any)
        if constraints_file.exists():
            with open(constraints_file) as f:
                all_constraints = json.load(f)
        else:
            all_constraints = {}

        # add/update this table's constraints
        all_constraints[table_name] = {
            "translator": self.to_dict(),
            "original_columns": original_columns,
        }

        # save back
        with open(constraints_file, "w") as f:
            json.dump(all_constraints, f, indent=2)

    @classmethod
    def load_metadata(cls, workspace_dir: Path, table_name: str) -> ConstraintTranslator | None:
        """load constraint metadata from workspace.

        Args:
            workspace_dir: Workspace directory path (ModelStore directory).
            table_name: Name of the table.

        Returns:
            ConstraintTranslator instance or None if not found.
        """
        constraints_file = workspace_dir / "constraints.json"

        if not constraints_file.exists():
            return None

        with open(constraints_file) as f:
            all_constraints = json.load(f)

        # get this table's constraints
        table_data = all_constraints.get(table_name)
        if not table_data:
            return None

        return cls.from_dict(table_data["translator"])

    @classmethod
    def load_original_columns(cls, workspace_dir: Path, table_name: str) -> list[str] | None:
        """load original column names for a table.

        Args:
            workspace_dir: Workspace directory path (ModelStore directory).
            table_name: Name of the table.

        Returns:
            List of original column names or None if not found.
        """
        constraints_file = workspace_dir / "constraints.json"

        if not constraints_file.exists():
            return None

        with open(constraints_file) as f:
            all_constraints = json.load(f)

        table_data = all_constraints.get(table_name)
        if not table_data:
            return None

        return table_data.get("original_columns")
