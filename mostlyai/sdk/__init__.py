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

import warnings
from importlib.metadata import distribution

try:
    extras = distribution("mostlyai").metadata.get_all("Provides-Extra") or []
    if "local-cpu" in extras:
        warnings.warn(
            "`local-cpu` extra is deprecated. Please install `mostlyai[local]` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
except Exception:
    pass

from mostlyai.sdk.client.api import MostlyAI

__all__ = ["MostlyAI"]
__version__ = "4.5.2"  # Do not set this manually. Use poetry version [params].
