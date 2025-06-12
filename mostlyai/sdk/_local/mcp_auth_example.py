from fastmcp import FastMCP, Context
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    jwks_uri="http://localhost:8000/jwks.json",
    issuer="http://localhost:8000/",
    audience="my-mcp-server"
)

mcp = FastMCP(name="MostlyAI MCP Server", auth=auth)

@mcp.tool(description="Echo the query")
async def echo(query: str, ctx: Context):
    return {"echo": query}

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8081)