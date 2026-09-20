"""Three-tier permission model and enforcement engine for MCP tools."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set


class PermissionTier(str, Enum):
    """Explicit permission hierarchy for autonomous tool execution.

    - READ_ONLY: Safe inspection, querying, metric calculation, and inference.
    - SAFE_WRITE: Sandboxed state mutations (creating splits, training candidate models).
    - HIGH_RISK: Operations requiring external human supervisor tokens (model promotion, deletion).
      NOTE: Phase 4 strictly forbids exposing any HIGH_RISK tools.
    """
    READ_ONLY = "READ_ONLY"
    SAFE_WRITE = "SAFE_WRITE"
    HIGH_RISK = "HIGH_RISK"

    @property
    def level(self) -> int:
        levels = {
            PermissionTier.READ_ONLY: 1,
            PermissionTier.SAFE_WRITE: 2,
            PermissionTier.HIGH_RISK: 3,
        }
        return levels[self]

    def can_access(self, required_tier: PermissionTier) -> bool:
        """Determines if this permission tier satisfies the required permission level."""
        return self.level >= required_tier.level


class PermissionRegistry:
    """Central registry tracking tool permission requirements and enforcement."""

    def __init__(self) -> None:
        self._registry: Dict[str, PermissionTier] = {}

    def register(self, tool_name: str, tier: PermissionTier) -> None:
        """Associates a tool name with an explicit permission tier."""
        self._registry[tool_name] = tier

    def get_tier(self, tool_name: str) -> PermissionTier:
        """Returns the permission tier required by a given tool."""
        if tool_name not in self._registry:
            raise KeyError(f"Tool '{tool_name}' is not registered in PermissionRegistry.")
        return self._registry[tool_name]

    def is_registered(self, tool_name: str) -> bool:
        return tool_name in self._registry

    def list_tools_by_tier(self, tier: PermissionTier) -> List[str]:
        """Lists all tool names assigned to a specific tier."""
        return [name for name, t in self._registry.items() if t == tier]

    def has_high_risk_tools(self) -> bool:
        """Safety invariant: Verifies whether any HIGH_RISK tools are exposed."""
        return any(t == PermissionTier.HIGH_RISK for t in self._registry.values())

    def check_permission(self, tool_name: str, caller_tier: PermissionTier) -> bool:
        """Verifies if the caller's permission tier can execute the target tool."""
        required = self.get_tier(tool_name)
        return caller_tier.can_access(required)


# Global singleton permission registry
permission_registry = PermissionRegistry()
