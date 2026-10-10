"""Tool modules, using the same contract as the project's ``--tool-module``.

A tool module defines ``TOOLS`` (Realtime function definitions) and an async
``execute_tool(name, arguments)``. The result may be a plain value or an object with
``output`` and ``create_response`` attributes (like the project's ``ToolResult``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolOutcome:
    output: str
    create_response: bool = True


class ToolRegistry:
    def __init__(self, module_names: tuple[str, ...] = ()) -> None:
        self.definitions: list[dict[str, Any]] = []
        self._executors: dict[str, Callable[[str, dict[str, Any]], Awaitable[Any]]] = {}
        self._defaults: dict[str, bool] = {}
        for module_name in module_names:
            module = import_module(module_name)
            default_create = bool(getattr(module, "CREATE_RESPONSE", True))
            for tool in module.TOOLS:
                name = tool["name"]
                if name in self._executors:
                    raise ValueError(f"Duplicate tool name {name!r} in {module_name}")
                self.definitions.append(tool)
                self._executors[name] = module.execute_tool
                self._defaults[name] = default_create
            logger.info("Loaded tools from %s: %s", module_name, [t["name"] for t in module.TOOLS])

    async def run(self, name: str, raw_arguments: str) -> ToolOutcome:
        executor = self._executors.get(name)
        if executor is None:
            return ToolOutcome(json.dumps({"error": f"Unknown tool: {name}"}))
        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            return ToolOutcome(json.dumps({"error": f"Invalid arguments for {name}: {exc}"}))
        try:
            result = await executor(name, arguments)
        except Exception as exc:  # report to the model so it can recover
            logger.exception("Tool %s failed", name)
            return ToolOutcome(json.dumps({"error": f"{name} failed: {exc}"}))
        create = getattr(result, "create_response", self._defaults[name])
        output = getattr(result, "output", result)
        return ToolOutcome(output if isinstance(output, str) else json.dumps(output, ensure_ascii=False), create)
