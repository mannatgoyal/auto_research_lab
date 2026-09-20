"""Structured error hierarchies and serialization for MCP tools."""

from __future__ import annotations

from typing import Any, Dict, Optional


class MCPToolError(Exception):
    """Base structured exception for all MCP tool operations."""

    def __init__(
        self,
        message: str,
        code: str = "TOOL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class MCPValidationError(MCPToolError):
    """Raised when input parameters fail schema validation or contain invalid values."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class MCPEngineError(MCPToolError):
    """Raised when a deterministic ML engine operation fails during execution."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="ENGINE_ERROR", details=details)


class MCPPermissionError(MCPToolError):
    """Raised when a caller attempts an operation exceeding their permission tier."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="PERMISSION_DENIED", details=details)


class MCPNotFoundError(MCPToolError):
    """Raised when an experiment result, dataset, or MLflow run cannot be found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="NOT_FOUND", details=details)


class MCPTrackingError(MCPToolError):
    """Raised when MLflow tracking or artifact logging encounters a failure."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="TRACKING_ERROR", details=details)


def format_error_dict(exc: Exception) -> Dict[str, Any]:
    """Formats any exception into a predictable, machine-readable structured error dictionary."""
    if isinstance(exc, MCPToolError):
        return {
            "success": False,
            "error": exc.to_dict(),
        }

    # Map standard Python errors
    code = "UNKNOWN_ERROR"
    if isinstance(exc, (ValueError, TypeError)):
        code = "VALIDATION_ERROR"
    elif isinstance(exc, FileNotFoundError):
        code = "NOT_FOUND"
    elif isinstance(exc, PermissionError):
        code = "PERMISSION_DENIED"

    return {
        "success": False,
        "error": {
            "code": code,
            "message": str(exc),
            "details": {},
        },
    }
