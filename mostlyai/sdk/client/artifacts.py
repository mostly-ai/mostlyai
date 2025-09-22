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


from mostlyai.sdk.client.base import (
    GET,
    PATCH,
    _MostlyBaseClient,
)
from mostlyai.sdk.domain import (
    Artifact,
    ArtifactPatchConfig,
)


class _MostlyArtifactsClient(_MostlyBaseClient):
    SECTION = ["artifacts"]

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve artifact metadata including the shareable URL where the artifact can be viewed.
        Unauthenticated access is allowed to enable public sharing of artifacts.

        Args:
            artifact_id: The unique identifier of the artifact.

        Returns:
            The retrieved Artifact object.

        Example for retrieving an artifact:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            art = mostly.artifacts.get("INSERT_YOUR_ARTIFACT_ID")
            art
            ```
        """
        response = self.request(verb=GET, path=[artifact_id], response_type=Artifact)
        return response

    def _update(
        self,
        artifact_id: str,
        config: ArtifactPatchConfig,
    ) -> Artifact:
        response = self.request(
            verb=PATCH,
            path=[artifact_id],
            json=config,
            exclude_none_in_json=True,
            response_type=Artifact,
        )
        return response
