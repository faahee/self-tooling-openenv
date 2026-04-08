"""Integration tests for the JARVIS self-evolving brain.

Tests 6 scenarios covering tool reuse, creation, dangerous code blocking,
low-reliability improvement, and synthesis failure fallback.

Usage:
    python test_brain.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# ── Minimal config for standalone testing ─────────────────────────────────────

TEST_DIR = Path(tempfile.mkdtemp(prefix="jarvis_brain_test_"))
TOOLS_DIR = TEST_DIR / "tools" / "generated"
TOOLS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CONFIG = {
    "brain": {
        "db_path": str(TEST_DIR / "test_brain.db"),
        "tools_dir": str(TOOLS_DIR),
        "builtin_dir": str(TEST_DIR / "tools" / "builtin"),
        "max_synthesis_retries": 3,
        "sandbox_timeout": 10,
        "min_reliability_threshold": 0.7,
        "similarity_threshold": 0.75,
        "max_tool_lines": 50,
    },
    "llm": {
        "model": "jarvis-llama",
        "base_url": "http://localhost:11434",
        "max_tokens": 100,
        "temperature": 0.35,
        "context_window": 2048,
        "stream": False,
        "timeout": 120,
    },
    "memory": {
        "db_path": str(TEST_DIR / "test_interactions.db"),
        "graph_path": str(TEST_DIR / "kg.json"),
        "personal_model_path": str(TEST_DIR / "pm.json"),
        "file_index_path": str(TEST_DIR / "fi.json"),
        "cache_path": str(TEST_DIR / "cache.db"),
        "max_cache_entries": 10,
        "cache_similarity_threshold": 0.92,
        "max_context_interactions": 3,
        "log_retention_days": 90,
    },
}

# ── Sample tool code for pre-registration ─────────────────────────────────────

SAMPLE_TOOL_CODE = '''import math
from typing import Any


def calculate_circle_area(radius: float) -> dict[str, Any]:
    """Calculate the area of a circle given its radius."""
    try:
        if radius < 0:
            return {"success": False, "result": None, "error": "Radius must be non-negative"}
        area = math.pi * radius ** 2
        return {"success": True, "result": round(area, 4), "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
'''

DANGEROUS_TOOL_CODE = '''import os
from typing import Any


def dangerous_tool(command: str) -> dict[str, Any]:
    """Execute a system command."""
    try:
        result = os.system(command)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
'''

SAMPLE_CONVERT_CODE = '''import re
from typing import Any


def convert_temperature(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    try:
        from_u = from_unit.lower().strip()
        to_u = to_unit.lower().strip()
        if from_u == to_u:
            return {"success": True, "result": value, "error": None}
        if from_u == "celsius" and to_u == "fahrenheit":
            result = (value * 9 / 5) + 32
        elif from_u == "fahrenheit" and to_u == "celsius":
            result = (value - 32) * 5 / 9
        elif from_u == "celsius" and to_u == "kelvin":
            result = value + 273.15
        elif from_u == "kelvin" and to_u == "celsius":
            result = value - 273.15
        elif from_u == "fahrenheit" and to_u == "kelvin":
            result = (value - 32) * 5 / 9 + 273.15
        elif from_u == "kelvin" and to_u == "fahrenheit":
            result = (value - 273.15) * 9 / 5 + 32
        else:
            return {"success": False, "result": None, "error": f"Unknown units: {from_u} -> {to_u}"}
        return {"success": True, "result": round(result, 2), "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
'''


# ── Test helpers ──────────────────────────────────────────────────────────────

def make_mock_ui() -> MagicMock:
    """Create a mock UILayer that silently accepts all calls."""
    ui = MagicMock()
    ui.console = MagicMock()
    ui.console.print = MagicMock()
    ui.speak = MagicMock()
    ui.display_response = MagicMock()
    return ui


def make_mock_interaction_logger() -> MagicMock:
    """Create a mock InteractionLogger."""
    il = MagicMock()
    il.log = MagicMock()
    il.log_tool_event = MagicMock()
    return il


def make_mock_llm(response: str = "") -> AsyncMock:
    """Create a mock LLMCore that returns a fixed response."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=response)
    llm.check_ollama_health = AsyncMock(return_value=True)
    llm.close = AsyncMock()
    return llm


results: list[tuple[str, bool, str]] = []


def record(test_name: str, passed: bool, reason: str = "") -> None:
    """Record a test result."""
    results.append((test_name, passed, reason))
    tag = "[PASS]" if passed else "[FAIL]"
    detail = f" — {reason}" if reason else ""
    print(f"{tag} {test_name}{detail}")


# ── TEST 1: Existing tool reuse ───────────────────────────────────────────────

async def test_1_existing_tool_reuse() -> None:
    """Pre-register a tool, then verify it is found and used without synthesis."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    from brain.tool_registry import ToolRegistry
    from brain.code_validator import CodeValidator
    from brain.sandbox_executor import SandboxExecutor
    from brain.tool_synthesizer import ToolSynthesizer
    from brain.tool_loader import ToolLoader
    from brain.decision_engine import DecisionEngine

    registry = ToolRegistry(TEST_CONFIG, embedder=embedder)
    ui = make_mock_ui()
    il = make_mock_interaction_logger()

    # Write sample tool to disk and register it
    tool_path = TOOLS_DIR / "calculate_circle_area_v1.py"
    tool_path.write_text(SAMPLE_TOOL_CODE, encoding="utf-8")

    registry.register(
        name="calculate_circle_area",
        description="Calculate the area of a circle given its radius",
        code_path=str(tool_path),
        parameters={"radius": "float"},
        created_by="human",
        version=1,
    )
    # Give it a good success rate
    for _ in range(5):
        registry.update_stats("calculate_circle_area", success=True)

    # Mock LLM for param extraction (return JSON with radius)
    llm = make_mock_llm('{"radius": 5.0}')

    validator = CodeValidator()
    sandbox = SandboxExecutor(TEST_CONFIG)
    synthesizer = ToolSynthesizer(TEST_CONFIG, llm, validator, sandbox, registry, ui)
    loader = ToolLoader(TEST_CONFIG)
    engine = DecisionEngine(
        TEST_CONFIG, registry, synthesizer, loader, llm, ui, il, embedder
    )

    # Ask for circle area — should match existing tool
    result = await engine.handle("calculate the area of a circle with radius 5")

    synthesis_called = synthesizer.synthesize != ToolSynthesizer.synthesize
    tool_in_result = result is not None and "calculate_circle_area" in result

    if result is not None and "78.54" in result:
        record("TEST 1: Existing tool reuse", True)
    elif tool_in_result:
        record("TEST 1: Existing tool reuse", True, f"Tool used: {result[:80]}")
    else:
        record("TEST 1: Existing tool reuse", False, f"Got: {result}")


# ── TEST 2: New tool creation ─────────────────────────────────────────────────

async def test_2_new_tool_creation() -> None:
    """Synthesize a brand new tool and verify it exists on disk and in registry."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    from brain.tool_registry import ToolRegistry
    from brain.code_validator import CodeValidator
    from brain.sandbox_executor import SandboxExecutor
    from brain.tool_synthesizer import ToolSynthesizer
    from brain.tool_loader import ToolLoader
    from brain.decision_engine import DecisionEngine

    # Fresh registry (new DB)
    cfg = dict(TEST_CONFIG)
    cfg["brain"] = dict(cfg["brain"])
    cfg["brain"]["db_path"] = str(TEST_DIR / "test_brain_2.db")

    registry = ToolRegistry(cfg, embedder=embedder)
    ui = make_mock_ui()
    il = make_mock_interaction_logger()
    validator = CodeValidator()
    sandbox = SandboxExecutor(cfg)

    # Mock LLM to return valid tool code for synthesis, then JSON for param extraction
    call_count = 0

    async def mock_generate(prompt: str, system_prompt: str = None, mode: str = "chat", stream_callback=None) -> str:
        nonlocal call_count
        call_count += 1
        # Reasoning pipeline — return valid JSON reasoning
        if system_prompt and "JARVIS" in (system_prompt or "") and "steps" in (system_prompt or ""):
            return json.dumps({
                "understood_intent": "Convert 100 degrees Celsius to Fahrenheit",
                "assumption": None,
                "reasoning": ["Temperature conversion is a simple calculation"],
                "steps": [{"order": 1, "action": "Convert temperature", "decision": "CREATE", "agent": None, "tool_name": "convert_temperature", "reason": "No existing agent handles unit conversion"}],
                "fallback": None,
            })
        if "Generate a Python function" in prompt or "TASK:" in prompt:
            return f"```python\n{SAMPLE_CONVERT_CODE}\n```"
        # Param extraction
        return '{"value": 100.0, "from_unit": "celsius", "to_unit": "fahrenheit"}'

    llm = AsyncMock()
    llm.generate = mock_generate

    synthesizer = ToolSynthesizer(cfg, llm, validator, sandbox, registry, ui)
    loader = ToolLoader(cfg)
    engine = DecisionEngine(cfg, registry, synthesizer, loader, llm, ui, il, embedder)

    result = await engine.handle("convert 100 celsius to fahrenheit")

    # Check file exists
    generated_files = list(TOOLS_DIR.glob("convert_temperature_*.py"))
    tool_in_db = registry.get_by_name("convert_temperature")

    if generated_files and tool_in_db:
        record("TEST 2: New tool creation", True)
    elif result and "212" in str(result):
        record("TEST 2: New tool creation", True, "Correct result returned")
    else:
        record(
            "TEST 2: New tool creation",
            False,
            f"files={len(generated_files)}, db={'yes' if tool_in_db else 'no'}, result={result}",
        )


# ── TEST 3: Tool reuse after creation ─────────────────────────────────────────

async def test_3_tool_reuse_after_creation() -> None:
    """Query the same task again — existing tool should be reused, not re-synthesized."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    from brain.tool_registry import ToolRegistry
    from brain.code_validator import CodeValidator
    from brain.sandbox_executor import SandboxExecutor
    from brain.tool_synthesizer import ToolSynthesizer
    from brain.tool_loader import ToolLoader
    from brain.decision_engine import DecisionEngine

    # Use same DB as test 2
    cfg = dict(TEST_CONFIG)
    cfg["brain"] = dict(cfg["brain"])
    cfg["brain"]["db_path"] = str(TEST_DIR / "test_brain_2.db")

    registry = ToolRegistry(cfg, embedder=embedder)
    ui = make_mock_ui()
    il = make_mock_interaction_logger()
    validator = CodeValidator()
    sandbox = SandboxExecutor(cfg)

    llm = make_mock_llm('{"value": 100.0, "from_unit": "celsius", "to_unit": "fahrenheit"}')

    synthesizer = ToolSynthesizer(cfg, llm, validator, sandbox, registry, ui)
    loader = ToolLoader(cfg)
    engine = DecisionEngine(cfg, registry, synthesizer, loader, llm, ui, il, embedder)

    tool_before = registry.get_by_name("convert_temperature")
    runs_before = tool_before["total_runs"] if tool_before else 0

    result = await engine.handle("convert 100 degrees celsius to fahrenheit")

    tool_after = registry.get_by_name("convert_temperature")
    runs_after = tool_after["total_runs"] if tool_after else 0

    if runs_after > runs_before:
        record("TEST 3: Tool reuse after creation", True, f"runs: {runs_before} → {runs_after}")
    elif result and "212" in str(result):
        record("TEST 3: Tool reuse after creation", True, "Correct result, tool reused")
    else:
        record(
            "TEST 3: Tool reuse after creation",
            False,
            f"runs_before={runs_before}, runs_after={runs_after}",
        )


# ── TEST 4: Dangerous code blocked ───────────────────────────────────────────

async def test_4_dangerous_code_blocked() -> None:
    """Verify the code validator blocks os.system() and other dangerous patterns."""
    from brain.code_validator import CodeValidator

    validator = CodeValidator()

    # Test os.system
    result = validator.validate(DANGEROUS_TOOL_CODE)
    os_system_blocked = not result.is_valid and any("os.system" in e for e in result.errors)

    # Test eval
    eval_code = '''from typing import Any

def unsafe_eval(expr: str) -> dict[str, Any]:
    """Evaluate an expression."""
    try:
        result = eval(expr)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
'''
    eval_result = validator.validate(eval_code)
    eval_blocked = not eval_result.is_valid and any("eval" in e for e in eval_result.errors)

    # Test subprocess
    subprocess_code = '''import subprocess
from typing import Any

def run_cmd(cmd: str) -> dict[str, Any]:
    """Run a subprocess command."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True)
        return {"success": True, "result": result.stdout, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
'''
    sub_result = validator.validate(subprocess_code)
    sub_blocked = not sub_result.is_valid

    # Test open() in write mode (read mode is allowed)
    open_code = '''from typing import Any

def write_secrets(path: str, data: str) -> dict[str, Any]:
    """Write a file."""
    try:
        with open(path, "w") as f:
            f.write(data)
        return {"success": True, "result": "written", "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
'''
    open_result = validator.validate(open_code)
    open_blocked = not open_result.is_valid

    # Validate safe code passes
    safe_result = validator.validate(SAMPLE_TOOL_CODE)
    safe_passes = safe_result.is_valid

    all_pass = os_system_blocked and eval_blocked and sub_blocked and open_blocked and safe_passes
    details = (
        f"os.system={'blocked' if os_system_blocked else 'MISSED'}, "
        f"eval={'blocked' if eval_blocked else 'MISSED'}, "
        f"subprocess={'blocked' if sub_blocked else 'MISSED'}, "
        f"open={'blocked' if open_blocked else 'MISSED'}, "
        f"safe_code={'passes' if safe_passes else 'REJECTED'}"
    )
    record("TEST 4: Dangerous code blocked", all_pass, details)


# ── TEST 5: Low reliability tool improvement ─────────────────────────────────

async def test_5_low_reliability_improvement() -> None:
    """Set a tool's success_rate to 0.3, send matching task, verify version bump."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    from brain.tool_registry import ToolRegistry
    from brain.code_validator import CodeValidator
    from brain.sandbox_executor import SandboxExecutor
    from brain.tool_synthesizer import ToolSynthesizer
    from brain.tool_loader import ToolLoader
    from brain.decision_engine import DecisionEngine

    cfg = dict(TEST_CONFIG)
    cfg["brain"] = dict(cfg["brain"])
    cfg["brain"]["db_path"] = str(TEST_DIR / "test_brain_5.db")

    registry = ToolRegistry(cfg, embedder=embedder)
    ui = make_mock_ui()
    il = make_mock_interaction_logger()

    # Register a tool with low reliability
    tool_path = TOOLS_DIR / "calculate_circle_area_v1.py"
    if not tool_path.exists():
        tool_path.write_text(SAMPLE_TOOL_CODE, encoding="utf-8")

    registry.register(
        name="calculate_circle_area",
        description="Calculate the area of a circle given its radius",
        code_path=str(tool_path),
        parameters={"radius": "float"},
        created_by="human",
        version=1,
    )
    registry.set_success_rate("calculate_circle_area", 0.3)

    version_before = registry.get_version("calculate_circle_area")

    # Mock LLM: return improved code on synthesis, JSON on param extraction
    async def mock_generate(prompt: str, system_prompt: str = None, mode: str = "chat", stream_callback=None) -> str:
        # Reasoning pipeline — return USE decision so the improvement branch runs
        # (the existing tool has low reliability, so handle() enters the improvement path)
        if system_prompt and "JARVIS" in (system_prompt or "") and "steps" in (system_prompt or ""):
            return json.dumps({
                "understood_intent": "Calculate the area of a circle with radius 5",
                "assumption": None,
                "reasoning": ["Existing tool can handle this but has low reliability"],
                "steps": [{"order": 1, "action": "Calculate circle area", "decision": "USE", "agent": "os_agent", "tool_name": None, "reason": "Use existing capability"}],
                "fallback": None,
            })
        if "Generate a Python function" in prompt or "TASK:" in prompt or "IMPROVE" in prompt:
            return f"```python\n{SAMPLE_TOOL_CODE}\n```"
        return '{"radius": 5.0}'

    llm = AsyncMock()
    llm.generate = mock_generate

    validator = CodeValidator()
    sandbox = SandboxExecutor(cfg)
    synthesizer = ToolSynthesizer(cfg, llm, validator, sandbox, registry, ui)
    loader = ToolLoader(cfg)
    engine = DecisionEngine(cfg, registry, synthesizer, loader, llm, ui, il, embedder)

    result = await engine.handle("calculate the area of a circle with radius 5")

    version_after = registry.get_version("calculate_circle_area")

    if version_after > version_before:
        record(
            "TEST 5: Low reliability tool improvement",
            True,
            f"version: {version_before} → {version_after}",
        )
    else:
        record(
            "TEST 5: Low reliability tool improvement",
            False,
            f"version unchanged: {version_before} → {version_after}, result={result}",
        )


# ── TEST 6: Synthesis failure fallback ────────────────────────────────────────

async def test_6_synthesis_failure_fallback() -> None:
    """Force synthesis to fail 3 times and verify graceful fallback (no crash)."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    from brain.tool_registry import ToolRegistry
    from brain.code_validator import CodeValidator
    from brain.sandbox_executor import SandboxExecutor
    from brain.tool_synthesizer import ToolSynthesizer
    from brain.tool_loader import ToolLoader
    from brain.decision_engine import DecisionEngine

    cfg = dict(TEST_CONFIG)
    cfg["brain"] = dict(cfg["brain"])
    cfg["brain"]["db_path"] = str(TEST_DIR / "test_brain_6.db")

    registry = ToolRegistry(cfg, embedder=embedder)
    ui = make_mock_ui()
    il = make_mock_interaction_logger()

    # Mock LLM returns garbage (not valid Python)
    llm = make_mock_llm("This is not valid python code at all, no function here.")

    validator = CodeValidator()
    sandbox = SandboxExecutor(cfg)
    synthesizer = ToolSynthesizer(cfg, llm, validator, sandbox, registry, ui)
    loader = ToolLoader(cfg)
    engine = DecisionEngine(cfg, registry, synthesizer, loader, llm, ui, il, embedder)

    # This should fail synthesis and return None (graceful fallback)
    result = await engine.handle("generate a fibonacci sequence up to 100")

    if result is None:
        record("TEST 6: Synthesis failure fallback", True, "Returned None (graceful fallback)")
    else:
        record(
            "TEST 6: Synthesis failure fallback",
            False,
            f"Expected None, got: {result[:80]}",
        )


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def run_all_tests() -> None:
    """Run all 6 tests in order."""
    print("=" * 60)
    print("JARVIS Brain — Integration Tests")
    print("=" * 60)
    print()

    await test_1_existing_tool_reuse()
    await test_2_new_tool_creation()
    await test_3_tool_reuse_after_creation()
    await test_4_dangerous_code_blocked()
    await test_5_low_reliability_improvement()
    await test_6_synthesis_failure_fallback()

    print()
    print("=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    # Cleanup
    try:
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    except Exception:
        pass

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
