"""MCP Client adapter strictly interacting with MCPServer."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.mcpserver import MCPServer
from mcp_servers.common.permissions import PermissionTier, permission_registry
from mcp_servers.ml_server import create_ml_server
from orchestrator.state import ResearchState, ToolCallRecord


class MCPToolClient:
    """Client adapter connecting the orchestrator to the MCP tool server.
    
    CRITICAL INVARIANT: This client connects exclusively via the MCP protocol
    interface (list_tools, call_tool). It does NOT import or call deterministic
    ML functions directly.
    """

    def __init__(
        self,
        server: Optional[MCPServer] = None,
        caller_tier: PermissionTier = PermissionTier.SAFE_WRITE,
    ):
        self.server = server or create_ml_server()
        self.caller_tier = caller_tier
        self._cached_tool_names: Optional[List[str]] = None

    async def list_available_tools(self) -> List[str]:
        """Queries the MCP server for all registered tools."""
        if self._cached_tool_names is None:
            tools = await self.server.list_tools()
            self._cached_tool_names = [t.name for t in tools]
        return self._cached_tool_names

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        state: Optional[ResearchState] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Executes a tool call over the MCP server, enforcing permissions and logging to state."""
        start_time = time.perf_counter()

        # 1. Verify tool availability
        available_tools = await self.list_available_tools()
        if tool_name not in available_tools:
            err_msg = f"Tool '{tool_name}' is not registered on the MCP server."
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            if state:
                state.add_tool_call(record)
            return False, {}, err_msg

        # 2. Enforce permission tier
        tool_tier = permission_registry.get_tier(tool_name)
        if tool_tier == PermissionTier.HIGH_RISK:
            err_msg = f"Security Violation: Tool '{tool_name}' is classified as HIGH_RISK and cannot be executed."
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            if state:
                state.add_tool_call(record)
            return False, {}, err_msg

        if not permission_registry.check_permission(tool_name, self.caller_tier):
            err_msg = f"Permission Denied: Caller tier '{self.caller_tier}' cannot execute '{tool_tier}' tool '{tool_name}'."
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            if state:
                state.add_tool_call(record)
            return False, {}, err_msg

        # 3. Dispatch through MCP server protocol
        try:
            res = await self.server.call_tool(tool_name, arguments)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract structured data
            data: Dict[str, Any] = {}
            if res.structured_content and "result" in res.structured_content:
                data = res.structured_content["result"]
            elif res.structured_content:
                data = res.structured_content
            elif res.content and len(res.content) > 0 and hasattr(res.content[0], "text"):
                data = json.loads(res.content[0].text)

            is_success = data.get("success", True) if isinstance(data, dict) else True
            err_str = None
            if not is_success and isinstance(data, dict) and "error" in data:
                err_info = data["error"]
                err_str = err_info.get("message", "Tool returned failure") if isinstance(err_info, dict) else str(err_info)

            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=data,
                success=is_success,
                duration_ms=duration_ms,
                error=err_str,
            )
            if state:
                state.add_tool_call(record)
            return is_success, data, err_str

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"MCP invocation exception on '{tool_name}': {e}"
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                duration_ms=duration_ms,
                error=err_msg,
            )
            if state:
                state.add_tool_call(record)
            return False, {}, err_msg
