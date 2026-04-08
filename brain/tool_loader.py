"""Tool Loader — Dynamic loading of synthesized tool functions via importlib."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Callable

logger = logging.getLogger(__name__)


class ToolLoader:
    """Dynamically loads Python tool functions from file paths at runtime.

    Uses importlib to load .py files and extract their main callable
    for execution by the Decision Engine.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the tool loader.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.tools_dir = Path(config["brain"]["tools_dir"])
        self._loaded: dict[str, Callable] = {}
        self._modules: dict[str, ModuleType] = {}

    def load(self, name: str, code_path: str) -> Callable | None:
        """Load a tool function from a Python file.

        Imports the file as a module and extracts the main function.
        If a function matching the tool name exists, it is used.
        Otherwise falls back to the first public callable in the module.

        Args:
            name: Tool name (used as module identifier).
            code_path: Path to the .py file.

        Returns:
            The tool's main callable, or None if loading failed.
        """
        path = Path(code_path)
        if not path.exists():
            logger.error("Tool file not found: %s", path)
            return None

        try:
            module_name = f"brain_tool_{name}"
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                logger.error("Cannot create module spec for: %s", path)
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Try to find a function matching the tool name first
            func = getattr(module, name, None)
            if func is not None and callable(func) and not isinstance(func, type):
                self._loaded[name] = func
                self._modules[name] = module
                logger.info("Loaded tool: %s from %s", name, path)
                return func

            # Fallback: find the first public callable function
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                if callable(attr) and not isinstance(attr, type):
                    self._loaded[name] = attr
                    self._modules[name] = module
                    logger.info(
                        "Loaded tool: %s (function: %s) from %s",
                        name,
                        attr_name,
                        path,
                    )
                    return attr

            logger.error("No callable function found in: %s", path)
            return None

        except Exception as exc:
            logger.error("Failed to load tool %s from %s: %s", name, path, exc)
            return None

    def load_all(self, registry) -> dict[str, Callable]:
        """Load all active tools from the registry.

        Args:
            registry: ToolRegistry instance.

        Returns:
            Dict mapping tool names to their callables.
        """
        tools = registry.get_all_active()
        loaded: dict[str, Callable] = {}
        for tool in tools:
            func = self.load(tool["name"], tool["code_path"])
            if func:
                loaded[tool["name"]] = func
        logger.info("Loaded %d/%d active tools from registry", len(loaded), len(tools))
        return loaded

    def get(self, name: str) -> Callable | None:
        """Get a previously loaded tool function by name.

        Args:
            name: Tool name.

        Returns:
            The callable or None if not loaded.
        """
        return self._loaded.get(name)

    def unload(self, name: str) -> None:
        """Remove a loaded tool from memory.

        Args:
            name: Tool name to unload.
        """
        self._loaded.pop(name, None)
        self._modules.pop(name, None)
        logger.info("Unloaded tool: %s", name)

    def is_loaded(self, name: str) -> bool:
        """Check if a tool is currently loaded in memory.

        Args:
            name: Tool name.

        Returns:
            True if the tool is loaded.
        """
        return name in self._loaded

    def reload(self, name: str, code_path: str) -> Callable | None:
        """Unload and reload a tool (used after version upgrade).

        Args:
            name: Tool name.
            code_path: Path to the new version's .py file.

        Returns:
            The freshly loaded callable, or None on failure.
        """
        self.unload(name)
        return self.load(name, code_path)
