"""Shared type aliases for the OpenAI/Hey_ JSON shapes."""

from typing import Any

JsonDict = dict[str, Any]
Message = dict[str, Any]
ToolChoice = str | JsonDict | None
HeyEvent = tuple[str, Any]
FinalParts = dict[str, Any | None]
