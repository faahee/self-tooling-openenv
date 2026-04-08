"""MCP Client — Model Context Protocol integration for JARVIS.

Manages connections to MCP servers (stdio and SSE transports),
discovers tools, and executes tool calls via JSON-RPC 2.0.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Represents a single MCP server connection."""

    name: str
    transport: str  # "stdio" or "sse"
    command: str = ""  # e.g. "npx" or "uvx"
    args: list[str] = field(default_factory=list)
    url: str = ""  # SSE endpoint URL
    env: dict[str, str] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    last_ping: float = 0.0
    # Internal — stdio transport
    _process: Any = field(default=None, repr=False)
    _request_id: int = field(default=0, repr=False)
    _sync_stdio: bool = field(default=False, repr=False)


class MCPClient:
    """Manages multiple MCP server connections and provides tool discovery/execution.

    Supports two transports:
      - stdio: spawns server as subprocess, communicates via stdin/stdout JSON-RPC
      - sse: connects to HTTP SSE endpoint (not yet implemented)

    Tool names are prefixed as {server_name}__{tool_name} to avoid conflicts.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.mcp_config: dict = config.get("mcp", {})
        self.enabled: bool = self.mcp_config.get("enabled", False)
        self.connection_timeout: int = self.mcp_config.get("connection_timeout", 30)
        self.tool_timeout: int = self.mcp_config.get("tool_timeout", 30)
        self.auto_reconnect: bool = self.mcp_config.get("auto_reconnect", True)
        self.servers: dict[str, MCPServer] = {}
        self._tool_index: dict[str, tuple[str, str]] = {}  # prefixed_name → (server_name, raw_tool_name)

    # ── Connection Management ─────────────────────────────────────────────────

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all enabled MCP servers defined in config.

        Returns:
            Dict of {server_name: connected_bool}.
        """
        if not self.enabled:
            logger.info("MCP disabled in config.")
            return {}

        server_defs = self.mcp_config.get("servers", {})
        results: dict[str, bool] = {}

        for name, sdef in server_defs.items():
            if not sdef.get("enabled", True):
                logger.info("MCP server '%s' disabled, skipping.", name)
                results[name] = False
                continue
            try:
                ok = await self.connect_server(name, sdef)
                results[name] = ok
            except Exception as exc:
                logger.error("Failed to connect MCP server '%s': %s", name, exc)
                results[name] = False

        total_tools = sum(len(s.tools) for s in self.servers.values() if s.connected)
        connected = sum(1 for v in results.values() if v)
        logger.info(
            "MCP: %d/%d servers connected, %d tools available.",
            connected, len(server_defs), total_tools,
        )
        return results

    async def connect_server(self, name: str, server_def: dict) -> bool:
        """Connect to a single MCP server.

        Args:
            name: Server name (used as tool prefix).
            server_def: Server definition dict from config.

        Returns:
            True if connected and tools discovered.
        """
        transport = server_def.get("transport", "stdio")
        server = MCPServer(
            name=name,
            transport=transport,
            command=server_def.get("command", ""),
            args=server_def.get("args", []),
            url=server_def.get("url", ""),
            env=server_def.get("env", {}),
        )

        if transport == "stdio":
            ok = await self._connect_stdio(server)
        else:
            logger.warning("SSE transport not yet implemented for '%s'.", name)
            return False

        if ok:
            self.servers[name] = server
            # Build tool index
            for tool in server.tools:
                prefixed = f"{name}__{tool['name']}"
                self._tool_index[prefixed] = (name, tool["name"])
            logger.info(
                "MCP server '%s' connected: %d tools.", name, len(server.tools)
            )
        return ok

    async def _connect_stdio(self, server: MCPServer) -> bool:
        """Spawn a subprocess and initialize the MCP server via JSON-RPC.

        Args:
            server: MCPServer instance to connect.

        Returns:
            True if initialization + tool discovery succeeded.
        """
        env = {**os.environ, **server.env}

        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    server.command,
                    *server.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                ),
                timeout=self.connection_timeout,
            )
            server._sync_stdio = False
        except asyncio.TimeoutError:
            logger.error("Timeout spawning MCP server '%s'.", server.name)
            return False
        except FileNotFoundError:
            logger.error(
                "MCP server '%s': command '%s' not found. Is it installed?",
                server.name, server.command,
            )
            return False
        except (NotImplementedError, PermissionError):
            logger.info(
                "Async subprocess transport unavailable; "
                "falling back to sync stdio for MCP server '%s'.",
                server.name,
            )
            try:
                process = await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.Popen,
                        [server.command, *server.args],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    ),
                    timeout=self.connection_timeout,
                )
                server._sync_stdio = True
            except FileNotFoundError:
                logger.error(
                    "MCP server '%s': command '%s' not found. Is it installed?",
                    server.name, server.command,
                )
                return False
            except asyncio.TimeoutError:
                logger.error("Timeout spawning MCP server '%s'.", server.name)
                return False

        server._process = process

        # JSON-RPC: initialize
        init_result = await self._jsonrpc_request(server, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "jarvis", "version": "1.0.0"},
        })

        if init_result is None:
            logger.error("MCP server '%s' initialize failed.", server.name)
            await self._kill_process(server)
            return False

        # Send initialized notification
        await self._jsonrpc_notify(server, "notifications/initialized", {})

        # Discover tools
        tools_result = await self._jsonrpc_request(server, "tools/list", {})
        if tools_result and "tools" in tools_result:
            server.tools = tools_result["tools"]
        else:
            server.tools = []

        server.connected = True
        server.last_ping = time.time()
        return True

    # ── Tool Execution ────────────────────────────────────────────────────────

    async def call_tool(
        self, prefixed_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a tool on the appropriate MCP server.

        Args:
            prefixed_name: Tool name in format "{server}__{tool}".
            arguments: Tool arguments dict.

        Returns:
            Result dict with 'success', 'result', 'error' keys (JARVIS standard format).
        """
        if prefixed_name not in self._tool_index:
            return {"success": False, "result": None, "error": f"Tool '{prefixed_name}' not found."}

        server_name, raw_name = self._tool_index[prefixed_name]
        server = self.servers.get(server_name)

        if not server or not server.connected:
            # Try reconnect
            if self.auto_reconnect and server_name in self.mcp_config.get("servers", {}):
                logger.info("Attempting reconnect to '%s'...", server_name)
                ok = await self.connect_server(
                    server_name, self.mcp_config["servers"][server_name]
                )
                if not ok:
                    return {"success": False, "result": None, "error": f"Server '{server_name}' disconnected."}
                server = self.servers[server_name]
            else:
                return {"success": False, "result": None, "error": f"Server '{server_name}' disconnected."}

        try:
            result = await asyncio.wait_for(
                self._jsonrpc_request(server, "tools/call", {
                    "name": raw_name,
                    "arguments": arguments or {},
                }),
                timeout=self.tool_timeout,
            )
        except asyncio.TimeoutError:
            return {"success": False, "result": None, "error": f"Tool '{raw_name}' timed out ({self.tool_timeout}s)."}

        if result is None:
            return {"success": False, "result": None, "error": f"Tool '{raw_name}' returned no result."}

        # MCP tools return content array — extract text
        content_parts = result.get("content", [])
        text_parts = [
            p.get("text", "") for p in content_parts if p.get("type") == "text"
        ]
        combined = "\n".join(text_parts) if text_parts else json.dumps(result)

        return {"success": True, "result": combined, "error": None}

    # ── Tool Discovery ────────────────────────────────────────────────────────

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Return all available tools from all connected servers.

        Returns:
            List of tool dicts with prefixed names and original schemas.
        """
        tools = []
        for server in self.servers.values():
            if not server.connected:
                continue
            for tool in server.tools:
                tools.append({
                    "name": f"{server.name}__{tool['name']}",
                    "server": server.name,
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                })
        return tools

    def get_tool_schemas_for_ollama(self) -> list[dict[str, Any]]:
        """Format MCP tools as Ollama-compatible tool schemas.

        Returns:
            List of tool schemas in Ollama/OpenAI function-calling format.
        """
        schemas = []
        for tool_info in self.get_all_tools():
            input_schema = tool_info.get("inputSchema", {})
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "parameters": {
                        "type": input_schema.get("type", "object"),
                        "properties": input_schema.get("properties", {}),
                        "required": input_schema.get("required", []),
                    },
                },
            })
        return schemas

    def find_tool(self, query: str) -> dict[str, Any] | None:
        """Find the best matching MCP tool for a natural language query.

        Uses keyword matching against tool names and descriptions.
        Returns the best match or None.

        Args:
            query: Natural language description of the desired tool.

        Returns:
            Tool info dict or None if no match found.
        """
        query_lower = query.lower()
        special_map = [
            (("powershell", "command"), "windows_mcp__PowerShell", 1.0),
            (("clipboard",), "windows_mcp__Clipboard", 1.0),
            (("notification",), "windows_mcp__Notification", 1.0),
            (("notify",), "windows_mcp__Notification", 0.95),
            (("process",), "windows_mcp__Process", 1.0),
            (("processes",), "windows_mcp__Process", 1.0),
            (("inspect", "screen"), "windows_mcp__Snapshot", 1.0),
            (("observe", "screen"), "windows_mcp__Snapshot", 1.0),
            (("switch", "window"), "windows_mcp__App", 0.95),
            (("focus", "window"), "windows_mcp__App", 0.95),
        ]
        tools_by_name = {tool["name"]: tool for tool in self.get_all_tools()}
        for needles, tool_name, score in special_map:
            if all(needle in query_lower for needle in needles) and tool_name in tools_by_name:
                return {**tools_by_name[tool_name], "match_score": score}

        query_words = set(query_lower.split())
        best_match: dict[str, Any] | None = None
        best_score = 0.0

        for tool_info in self.get_all_tools():
            name = tool_info["name"].replace("__", " ").replace("_", " ").lower()
            desc = tool_info.get("description", "").lower()
            searchable = f"{name} {desc}"
            searchable_words = set(searchable.split())

            # Word overlap score
            overlap = len(query_words & searchable_words)
            if overlap == 0:
                continue

            score = overlap / max(len(query_words), 1)

            # Boost exact substring matches in name
            raw_name = tool_info["name"].split("__")[-1].lower()
            for word in query_words:
                if word in raw_name:
                    score += 0.3

            if score > best_score:
                best_score = score
                best_match = {**tool_info, "match_score": round(score, 2)}

        if best_match and best_score >= 0.3:
            return best_match
        return None

    # ── Health & Lifecycle ────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, str]:
        """Ping all servers and return their status.

        Returns:
            Dict of {server_name: "ok" | "disconnected" | "error"}.
        """
        results: dict[str, str] = {}
        for name, server in self.servers.items():
            if not server.connected or not server._process:
                results[name] = "disconnected"
                continue
            if server._process.returncode is not None:
                server.connected = False
                results[name] = "crashed"
                continue
            server.last_ping = time.time()
            results[name] = "ok"
        return results

    async def disconnect_all(self) -> None:
        """Gracefully shut down all MCP server connections."""
        for name, server in self.servers.items():
            try:
                await self._kill_process(server)
                server.connected = False
                logger.info("MCP server '%s' disconnected.", name)
            except Exception as exc:
                logger.warning("Error disconnecting '%s': %s", name, exc)
        self.servers.clear()
        self._tool_index.clear()

    # ── JSON-RPC Protocol ─────────────────────────────────────────────────────

    async def _jsonrpc_request(
        self, server: MCPServer, method: str, params: dict
    ) -> dict | None:
        """Send a JSON-RPC 2.0 request and wait for the response.

        Args:
            server: Target MCP server.
            method: JSON-RPC method name.
            params: Method parameters.

        Returns:
            Result dict from the response, or None on error.
        """
        if not server._process or not server._process.stdin or not server._process.stdout:
            return None

        server._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": server._request_id,
            "method": method,
            "params": params,
        }

        try:
            msg = json.dumps(request) + "\n"
            await self._write_process(server, msg.encode())

            # Read response line
            raw = await asyncio.wait_for(
                self._read_process_line(server),
                timeout=self.connection_timeout,
            )
            if not raw:
                return None

            response = json.loads(raw.decode().strip())

            if "error" in response:
                logger.error(
                    "MCP '%s' error on '%s': %s",
                    server.name, method, response["error"],
                )
                return None

            return response.get("result")

        except asyncio.TimeoutError:
            logger.error("MCP '%s' timeout on '%s'.", server.name, method)
            return None
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("MCP '%s' protocol error: %s", server.name, exc)
            return None

    async def _jsonrpc_notify(
        self, server: MCPServer, method: str, params: dict
    ) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected).

        Args:
            server: Target MCP server.
            method: Notification method name.
            params: Notification parameters.
        """
        if not server._process or not server._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            msg = json.dumps(notification) + "\n"
            await self._write_process(server, msg.encode())
        except OSError as exc:
            logger.warning("MCP '%s' notify error: %s", server.name, exc)

    async def _kill_process(self, server: MCPServer) -> None:
        """Terminate a server subprocess gracefully."""
        if server._process and server._process.returncode is None:
            try:
                server._process.terminate()
                if server._sync_stdio:
                    await asyncio.wait_for(
                        asyncio.to_thread(server._process.wait, 5),
                        timeout=6,
                    )
                else:
                    await asyncio.wait_for(server._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                server._process.kill()
            except subprocess.TimeoutExpired:
                server._process.kill()
            except ProcessLookupError:
                pass
        server._process = None
        server._sync_stdio = False

    async def _write_process(self, server: MCPServer, payload: bytes) -> None:
        """Write bytes to an MCP server process regardless of subprocess type."""
        if server._sync_stdio:
            await asyncio.to_thread(self._sync_write, server._process.stdin, payload)
            return
        server._process.stdin.write(payload)
        await server._process.stdin.drain()

    async def _read_process_line(self, server: MCPServer) -> bytes:
        """Read a single line from an MCP server process regardless of subprocess type."""
        if server._sync_stdio:
            return await asyncio.to_thread(server._process.stdout.readline)
        return await server._process.stdout.readline()

    @staticmethod
    def _sync_write(handle: Any, payload: bytes) -> None:
        """Blocking stdio write for subprocess.Popen handles."""
        handle.write(payload)
        handle.flush()
