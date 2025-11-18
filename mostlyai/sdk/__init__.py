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

"""Synthetic Data SDK - Python toolkit for high-fidelity, privacy-safe synthetic data.

The SDK supports two operating modes:
- LOCAL mode: Train and generate synthetic data locally on your own compute resources
- CLIENT mode: Connect to a remote MOSTLY AI platform for training & generation

Key Resources
-------------
The SDK manages four core resources:

1. **Generators** - Train synthetic data generators on tabular or language data
2. **Synthetic Datasets** - Create synthetic samples from trained generators
3. **Connectors** - Connect to data sources (databases, cloud storage)
4. **Datasets** - Create datasets with instructions (CLIENT mode only)

Core Operations
---------------
- `mostly.train(config)` - Train a generator on tabular or language data
- `mostly.generate(g, config)` - Generate synthetic data records
- `mostly.probe(g, config)` - Live probe the generator on demand
- `mostly.connect(config)` - Connect to external data sources

Key Features
------------
- Broad data support (mixed-type, single/multi-table, time-series)
- Multiple model types (TabularARGN, Hugging Face LLMs, LSTM)
- Advanced training options (GPU/CPU, Differential Privacy)
- Automated quality assurance with metrics and HTML reports
- Flexible sampling (up-sampling, conditional simulation, rebalancing)
- Seamless integration with external data sources

For more information, visit: https://mostly-ai.github.io/mostlyai/
"""

from mostlyai.sdk.client.api import MostlyAI

__all__ = ["MostlyAI"]
__version__ = "5.7.0"  # Do not set this manually. Use poetry version [params].
