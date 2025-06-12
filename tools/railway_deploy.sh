#!/bin/bash
set -e

uv run mcp-server --transport "streamable-http" --host "0.0.0.0" --port "8000"
