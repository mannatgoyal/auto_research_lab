"""Audit logging framework capturing all MCP tool executions with sanitized parameters."""

from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp_servers.common.permissions import PermissionTier

SENSITIVE_KEY_PATTERNS = {"token", "secret", "password", "key", "auth", "credential"}


def sanitize_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redacts values for keys matching sensitive patterns."""
    sanitized = {}
    for k, v in args.items():
        if any(pat in k.lower() for pat in SENSITIVE_KEY_PATTERNS):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_arguments(v)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            sanitized[k] = v
        elif isinstance(v, (list, tuple)):
            # Cap long lists (e.g. large arrays) for audit log readability
            if len(v) > 50:
                sanitized[k] = f"[{len(v)} items: {str(v[:3])[:-1]}...]"
            else:
                sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized


@dataclass
class AuditRecord:
    """Immutable audit entry recording a single tool invocation."""

    timestamp: str
    tool_name: str
    permission_tier: str
    sanitized_input: Dict[str, Any]
    success: bool
    duration_ms: float
    entity_id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """Thread-safe logger recording MCP tool invocations to disk and in-memory buffer."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path or os.path.join("logs", "mcp_audit.jsonl")
        self._records: List[AuditRecord] = []
        self._lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(
        self,
        tool_name: str,
        tier: PermissionTier | str,
        input_args: Dict[str, Any],
        success: bool,
        duration_ms: float,
        entity_id: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Records an execution event to both in-memory history and the audit file."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tier_str = tier.value if isinstance(tier, PermissionTier) else str(tier)
        clean_args = sanitize_arguments(input_args)

        record = AuditRecord(
            timestamp=now,
            tool_name=tool_name,
            permission_tier=tier_str,
            sanitized_input=clean_args,
            success=success,
            duration_ms=round(duration_ms, 3),
            entity_id=entity_id,
            error=error,
        )

        with self._lock:
            self._records.append(record)
            # Retain maximum 2000 in-memory records
            if len(self._records) > 2000:
                self._records.pop(0)

            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(record.to_json() + "\n")
            except Exception:
                pass

        return record

    def get_recent(self, limit: int = 10) -> List[AuditRecord]:
        """Retrieves most recent audit records from memory."""
        with self._lock:
            return list(self._records[-limit:])

    def clear(self) -> None:
        """Clears in-memory audit records."""
        with self._lock:
            self._records.clear()


# Global audit logger instance
audit_logger = AuditLogger()
