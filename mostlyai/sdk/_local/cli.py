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
import warnings
from pathlib import Path

import typer

from mostlyai.sdk._local.execution.jobs import execute_generation_job, execute_training_job

cli = typer.Typer(pretty_exceptions_enable=False)


@cli.command()
def run_training(generator_id: str, home_dir: Path):
    # suppress any deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        execute_training_job(generator_id, home_dir)


@cli.command()
def run_generation(synthetic_dataset_id: str, home_dir: Path):
    # suppress any deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        execute_generation_job(synthetic_dataset_id, home_dir)


def run_cli():
    cli(standalone_mode=False)


if __name__ == "__main__":
    run_cli()
