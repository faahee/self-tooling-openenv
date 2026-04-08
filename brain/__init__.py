"""Brain — Self-evolving tool synthesis and management for JARVIS.

Provides autonomous tool creation, validation, sandboxed execution,
and registry management so JARVIS can learn new capabilities at runtime.
"""
from brain.code_validator import CodeValidator
from brain.decision_engine import DecisionEngine
from brain.sandbox_executor import SandboxExecutor
from brain.tool_loader import ToolLoader
from brain.tool_registry import ToolRegistry
from brain.tool_synthesizer import ToolSynthesizer

__all__ = [
    "CodeValidator",
    "DecisionEngine",
    "SandboxExecutor",
    "ToolLoader",
    "ToolRegistry",
    "ToolSynthesizer",
]
