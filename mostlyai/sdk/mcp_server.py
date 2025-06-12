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

"""Simple MCP Server with Keycloak OAuth Authentication."""

import logging
import os
import secrets
import time
from typing import Any, Literal

import click
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp.server import FastMCP
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

logger = logging.getLogger(__name__)


class KeycloakServerSettings(BaseSettings):
    """Settings for the simple Keycloak MCP server."""

    model_config = SettingsConfigDict(env_prefix="MCP_KEYCLOAK_")

    # Server settings
    host: str = "localhost"
    port: int = 8000
    server_url: AnyHttpUrl | None = None

    # Keycloak OAuth settings - MUST be provided via environment variables
    keycloak_client_id: str  # MCP_KEYCLOAK_KEYCLOAK_CLIENT_ID env var
    keycloak_client_secret: str  # MCP_KEYCLOAK_KEYCLOAK_CLIENT_SECRET env var
    keycloak_realm: str  # MCP_KEYCLOAK_KEYCLOAK_REALM env var
    keycloak_server_url: AnyHttpUrl = AnyHttpUrl(f"{os.environ['MOSTLY_BASE_URL']}/auth")
    keycloak_callback_path: str | None = None

    # MCP OAuth settings
    mcp_scope: str = "user"
    keycloak_scope: str = "openid"

    def __init__(self, **data):
        """Initialize settings with values from environment variables.

        Note: keycloak_client_id, keycloak_client_secret, keycloak_realm, and keycloak_server_url
        are required and must be provided via environment variables.
        """
        super().__init__(**data)
        self.server_url = self.server_url or AnyHttpUrl(
            f"{os.getenv('RAILWAY_SERVER_URL', f'http://{self.host}:{self.port}')}"
        )
        self.keycloak_callback_path = self.keycloak_callback_path or f"{self.server_url._url}keycloak/callback"

    @property
    def keycloak_auth_url(self) -> str:
        """Get the Keycloak authorization URL."""
        return f"{self.keycloak_server_url}/realms/{self.keycloak_realm}/protocol/openid-connect/auth"

    @property
    def keycloak_token_url(self) -> str:
        """Get the Keycloak token URL."""
        return f"{self.keycloak_server_url}/realms/{self.keycloak_realm}/protocol/openid-connect/token"


class KeycloakOAuthProvider(OAuthAuthorizationServerProvider):
    """Keycloak OAuth provider with essential functionality."""

    def __init__(self, settings: KeycloakServerSettings):
        self.settings = settings
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}
        self.state_mapping: dict[str, dict[str, str]] = {}
        # Store Keycloak tokens with MCP tokens using the format:
        # {"mcp_token": "keycloak_token"}
        self.token_mapping: dict[str, str] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Get OAuth client information."""
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull):
        """Register a new OAuth client."""
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Generate an authorization URL for Keycloak OAuth flow."""
        state = params.state or secrets.token_hex(16)

        # Store the state mapping
        self.state_mapping[state] = {
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "redirect_uri_provided_explicitly": str(params.redirect_uri_provided_explicitly),
            "client_id": client.client_id,
        }

        # Build Keycloak authorization URL
        auth_url = (
            f"{self.settings.keycloak_auth_url}"
            f"?client_id={self.settings.keycloak_client_id}"
            f"&redirect_uri={self.settings.keycloak_callback_path}"
            f"&scope={self.settings.keycloak_scope}"
            f"&response_type=code"
            f"&state={state}"
        )

        return auth_url

    async def handle_keycloak_callback(self, code: str, state: str) -> str:
        """Handle Keycloak OAuth callback."""
        state_data = self.state_mapping.get(state)
        if not state_data:
            raise HTTPException(400, "Invalid state parameter")

        redirect_uri = state_data["redirect_uri"]
        code_challenge = state_data["code_challenge"]
        redirect_uri_provided_explicitly = state_data["redirect_uri_provided_explicitly"] == "True"
        client_id = state_data["client_id"]

        # Exchange code for token with Keycloak
        async with create_mcp_http_client() as client:
            response = await client.post(
                self.settings.keycloak_token_url,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.settings.keycloak_client_id,
                    "client_secret": self.settings.keycloak_client_secret,
                    "code": code,
                    "redirect_uri": self.settings.keycloak_callback_path,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                raise HTTPException(400, f"Failed to exchange code for token: {response.text}")

            data = response.json()
            if "error" in data:
                raise HTTPException(400, data.get("error_description", data["error"]))

            keycloak_access_token = data["access_token"]
            keycloak_refresh_token = data["refresh_token"]

            # Create MCP authorization code
            new_code = f"mcp_{secrets.token_hex(16)}"
            self.auth_codes[new_code] = AuthorizationCode(
                code=new_code,
                client_id=client_id,
                redirect_uri=AnyUrl(redirect_uri),
                redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
                expires_at=time.time() + 300,  # 5 minutes
                scopes=[self.settings.mcp_scope],
                code_challenge=code_challenge,
            )

            # Store Keycloak token - we'll map the MCP token to this later
            self.access_tokens[keycloak_access_token] = AccessToken(
                token=keycloak_access_token,
                client_id=client_id,
                scopes=self.settings.keycloak_scope.split(),
                expires_at=None,
            )

            self.refresh_tokens[keycloak_refresh_token] = RefreshToken(
                token=keycloak_refresh_token,
                client_id=client_id,
                scopes=self.settings.keycloak_scope.split(),
                expires_at=None,
            )

        del self.state_mapping[state]
        return construct_redirect_uri(redirect_uri, code=new_code, state=state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        """Load an authorization code."""
        return self.auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        """Exchange authorization code for tokens."""
        if authorization_code.code not in self.auth_codes:
            raise ValueError("Invalid authorization code")

        # Generate MCP access token
        mcp_token = f"mcp_{secrets.token_hex(32)}"

        # Store MCP token
        self.access_tokens[mcp_token] = AccessToken(
            token=mcp_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + 3600,
        )

        # Find Keycloak token for this client
        keycloak_token = next(
            (
                token
                for token, data in self.access_tokens.items()
                if not token.startswith("mcp_") and data.client_id == client.client_id
            ),
            None,
        )

        # Store mapping between MCP token and Keycloak token
        if keycloak_token:
            self.token_mapping[mcp_token] = keycloak_token

        del self.auth_codes[authorization_code.code]

        return OAuthToken(
            access_token=mcp_token,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Load and validate an access token."""
        access_token = self.access_tokens.get(token)
        if not access_token:
            return None

        # Check if expired
        if access_token.expires_at and access_token.expires_at < time.time():
            del self.access_tokens[token]
            return None

        return access_token

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        """Load a refresh token."""
        refresh_token_obj = self.refresh_tokens.get(refresh_token)
        if not refresh_token_obj:
            return None

        # Check if expired
        if refresh_token_obj.expires_at and refresh_token_obj.expires_at < time.time():
            del self.refresh_tokens[refresh_token]
            return None

        return refresh_token_obj

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Exchange refresh token"""
        raise NotImplementedError("Not supported yet.")

    async def revoke_token(self, token: str | AccessToken | RefreshToken) -> None:
        """Revoke a token."""
        if token in self.access_tokens:
            del self.access_tokens[token]
        if token in self.token_mapping:
            del self.token_mapping[token]
        # TODO: Revoke refresh token


def create_keycloak_mcp_server(settings: KeycloakServerSettings) -> FastMCP:
    """Create a FastMCP server with Keycloak OAuth."""
    oauth_provider = KeycloakOAuthProvider(settings)

    auth_settings = AuthSettings(
        issuer_url=settings.server_url,
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[settings.mcp_scope],
            default_scopes=[settings.mcp_scope],
        ),
        required_scopes=[settings.mcp_scope],
    )

    # Create FastMCP server with OAuth
    app = FastMCP(
        name="mostlyai-mcp",
        description="Mostly AI MCP server with Keycloak OAuth authentication",
        auth_server_provider=oauth_provider,
        host=settings.host,
        port=settings.port,
        debug=True,
        auth=auth_settings,
    )

    @app.custom_route("/keycloak/callback", methods=["GET"])
    async def keycloak_callback_handler(request: Request) -> Response:
        """Handle Keycloak OAuth callback."""
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")

        if error:
            error_description = request.query_params.get("error_description", error)
            return JSONResponse({"error": error, "error_description": error_description}, status_code=400)

        if not code or not state:
            return JSONResponse({"error": "Missing code or state parameter"}, status_code=400)

        try:
            redirect_url = await oauth_provider.handle_keycloak_callback(code, state)
            return RedirectResponse(url=redirect_url, status_code=302)
        except HTTPException as e:
            return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)
        except Exception:
            logger.exception("Error in Keycloak callback")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    def get_keycloak_token() -> str:
        """Get the Keycloak token for the current request."""
        mcp_token = get_access_token()
        if not mcp_token:
            raise ValueError("No access token available")

        keycloak_token = oauth_provider.token_mapping.get(mcp_token.token)
        if not keycloak_token:
            raise ValueError("No Keycloak token found for this session")

        return keycloak_token

    @app.tool(description="Get the user info from Mostly AI.")
    async def get_user_info() -> dict[str, Any]:
        """Get the user info from Mostly AI."""
        keycloak_token = get_keycloak_token()

        async with create_mcp_http_client() as client:
            response = await client.get(
                f"{os.environ['MOSTLY_BASE_URL']}/api/v2/users/me",
                headers={"Authorization": f"Bearer {keycloak_token}"},
            )

            if response.status_code != 200:
                raise HTTPException(400, f"Failed to get user info: {response.text}")

            return response.json()

    @app.tool(description="List the connectors available to the user.")
    async def list_connectors() -> list[dict[str, Any]]:
        """List the connectors available to the user."""
        keycloak_token = get_keycloak_token()

        async with create_mcp_http_client() as client:
            response = await client.get(
                f"{os.environ['MOSTLY_BASE_URL']}/api/v2/connectors",
                headers={"Authorization": f"Bearer {keycloak_token}"},
            )

            if response.status_code != 200:
                raise HTTPException(400, f"Failed to list generators: {response.text}")

            return response.json()

    return app


@click.command()
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--host", default="localhost", help="Host to bind to")
@click.option(
    "--transport",
    default="sse",
    type=click.Choice(["sse", "streamable-http"]),
    help="Transport protocol to use ('sse' or 'streamable-http')",
)
def main(port: int, host: str, transport: Literal["sse", "streamable-http"]) -> int:
    """Run the simple Keycloak MCP server."""
    try:
        settings = KeycloakServerSettings(host=host, port=port)
        app = create_keycloak_mcp_server(settings)
        app.run(transport=transport)
        return 0
    except Exception as e:
        logger.exception("Failed to start server")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    exit(main())
