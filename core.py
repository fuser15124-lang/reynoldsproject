"""Pure, testable utilities for NOVA. No network or Streamlit dependency."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize_digits(value: str) -> str:
    """Convert Persian and Arabic-Indic digits to ASCII digits."""
    return value.translate(_DIGIT_MAP)


def requested_page_number(query: str) -> Optional[int]:
    """Extract an explicitly requested physical page number from Persian/English text."""
    q = normalize_digits(query.casefold())
    for pattern in (r"(?:صفحه(?:ٔ|ی)?|page|p\.)\s*(\d+)", r"(\d+)\s*(?:صفحه|page)"):
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return None


def make_signature(files: Iterable[Any]) -> tuple:
    """Stable upload signature based on filename, size and content (when available)."""
    rows = []
    for file in files or []:
        name = getattr(file, "name", "unnamed")
        try:
            data = file.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            size = len(data)
        except Exception:
            size = getattr(file, "size", 0)
            digest = ""
        rows.append((name, size, digest))
    return tuple(sorted(rows))


def export_conversation(history: list[dict], fmt: str = "markdown", language: str = "فارسی") -> str:
    """Serialize chat history as Markdown, plain text, or JSON."""
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps({"schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
                           "messages": history}, ensure_ascii=False, indent=2)
    lines = []
    for item in history:
        role = item.get("role", "unknown")
        if language == "فارسی":
            role = "شما" if role == "user" else ("دستیار" if role == "assistant" else role)
        else:
            role = role.title()
        content = str(item.get("content", ""))
        if fmt == "txt":
            lines.append(f"{role}\n{'=' * len(role)}\n{content}")
        else:
            lines.append(f"## {role}\n\n{content}")
    separator = "\n\n" if fmt == "txt" else "\n\n---\n\n"
    return separator.join(lines)
