"""Quick smoke test for the local Windows-MCP server."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mcp_client import MCPClient


CONFIG = {
    "mcp": {
        "enabled": True,
        "connection_timeout": 20,
        "tool_timeout": 20,
        "auto_reconnect": False,
        "servers": {
            "windows_mcp": {
                "enabled": True,
                "transport": "stdio",
                "command": r".\.venv\Scripts\python.exe",
                "args": ["-m", "windows_mcp"],
                "env": {
                    "ANONYMIZED_TELEMETRY": "false",
                },
            }
        },
    }
}


async def main() -> None:
    client = MCPClient(CONFIG)
    server_def = CONFIG["mcp"]["servers"]["windows_mcp"]
    connected = await client.connect_server("windows_mcp", server_def)
    print(f"connected={connected}")
    if not connected:
        return

    tools = client.get_all_tools()
    print(f"tool_count={len(tools)}")
    print("first_tools=" + json.dumps([tool["name"] for tool in tools[:10]]))

    result = await client.call_tool(
        "windows_mcp__PowerShell",
        {"command": 'Write-Output "windows-mcp-ok"', "timeout": 10},
    )
    print("powershell_success=" + str(result.get("success")))
    print("powershell_result=" + json.dumps(str(result.get("result", ""))[:200]))

    await client.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
