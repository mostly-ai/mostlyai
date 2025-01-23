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

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class APIError(Exception):
    def __init__(self, message: str = None, do_rich_print: bool = True):
        super().__init__(message)
        self.message = message
        if do_rich_print:
            console = Console()
            error_message = Text(self.message, style="bold red")
            error_panel = Panel(error_message, expand=False)
            console.print(error_panel)

    def __str__(self):
        return self.message


class APIStatusError(APIError):
    pass
