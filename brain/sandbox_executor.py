"""Sandbox Executor — Safe isolated execution environment for synthesized tools."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """Executes tool code in an isolated subprocess with timeout.

    Process-level isolation on Windows 11: the tool runs in a completely
    separate Python process with no access to the parent's memory space.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.timeout: int = config["brain"]["sandbox_timeout"]
        self.python_exe: str = sys.executable

    async def execute(
        self,
        code: str,
        function_name: str,
        test_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tool code in a sandboxed subprocess with dummy parameters.

        The code is written to a temporary file with a test harness appended,
        then executed in a fresh Python process with a hard timeout.

        Args:
            code: The complete Python tool source code.
            function_name: Name of the function to call.
            test_params: Test parameters to pass to the function.

        Returns:
            Dict with keys: passed (bool), result (any), error (str|None).
        """
        harness = self._build_test_harness(code, function_name, test_params)

        tmp_file: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
                dir=tempfile.gettempdir(),
            ) as f:
                f.write(harness)
                tmp_file = Path(f.name)

            try:
                process = await asyncio.create_subprocess_exec(
                    self.python_exe,
                    str(tmp_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tempfile.gettempdir(),
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
                return {
                    "passed": False,
                    "result": None,
                    "error": f"Sandbox timeout: execution exceeded {self.timeout}s",
                }

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            if process.returncode != 0:
                # Check stdout first — harness prints JSON errors there
                if stdout_text:
                    try:
                        last_line = stdout_text.splitlines()[-1]
                        result = json.loads(last_line)
                        if "error" in result:
                            return {
                                "passed": False,
                                "result": None,
                                "error": result["error"],
                            }
                    except (json.JSONDecodeError, IndexError):
                        pass
                error_msg = (
                    stderr_text[:500]
                    if stderr_text
                    else f"Process exited with code {process.returncode}"
                )
                return {"passed": False, "result": None, "error": error_msg}

            if not stdout_text:
                return {
                    "passed": False,
                    "result": None,
                    "error": "No output from sandbox process",
                }

            # Parse the last line as JSON (the harness prints the result there)
            try:
                last_line = stdout_text.splitlines()[-1]
                result = json.loads(last_line)
            except (json.JSONDecodeError, IndexError):
                return {
                    "passed": False,
                    "result": None,
                    "error": f"Invalid sandbox output: {stdout_text[:200]}",
                }

            if result.get("sandbox_pass"):
                return {
                    "passed": True,
                    "result": result.get("result"),
                    "error": None,
                }
            return {
                "passed": False,
                "result": None,
                "error": result.get("error", "Unknown sandbox failure"),
            }

        except OSError as exc:
            logger.error("Sandbox OS error: %s", exc)
            return {"passed": False, "result": None, "error": f"Sandbox OS error: {exc}"}
        finally:
            if tmp_file and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    def _build_test_harness(
        self,
        code: str,
        function_name: str,
        test_params: dict[str, Any],
    ) -> str:
        """Build a self-contained test script that runs the tool safely.

        The harness:
        1. Defines the tool function (inline code)
        2. Calls it with dummy parameters
        3. Validates the return format
        4. Prints structured JSON result to stdout

        Args:
            code: Tool source code.
            function_name: Function to call.
            test_params: Parameters to pass.

        Returns:
            Complete Python script string.
        """
        params_json = json.dumps(test_params)

        harness = (
            "import json\n"
            "import sys\n"
            "\n"
            "# ---- Tool code (under test) ----\n"
            f"{code}\n"
            "# ---- End tool code ----\n"
            "\n"
            "# ---- Sandbox test harness ----\n"
            "try:\n"
            f"    _params = json.loads({repr(params_json)})\n"
            f"    _result = {function_name}(**_params)\n"
            "    if not isinstance(_result, dict):\n"
            '        print(json.dumps({"sandbox_pass": False, "error": "Return value must be a dict"}))\n'
            "        sys.exit(1)\n"
            '    for _key in ("success", "result", "error"):\n'
            "        if _key not in _result:\n"
            '            print(json.dumps({"sandbox_pass": False, "error": f"Missing required key: {_key}"}))\n'
            "            sys.exit(1)\n"
            '    print(json.dumps({"sandbox_pass": True, "result": _result}))\n'
            "except Exception as _e:\n"
            '    print(json.dumps({"sandbox_pass": False, "error": str(_e)}))\n'
            "    sys.exit(1)\n"
        )
        return harness

    @staticmethod
    def check_result_quality(
        result: dict[str, Any],
        blueprint: Any,
    ) -> tuple[bool, str]:
        """Check that sandbox result meets quality expectations.

        Validates that the returned dict has the expected structure,
        the success field is boolean, and key fields are not empty.

        Args:
            result: The dict returned by the tool function.
            blueprint: ToolBlueprint with expected return field names.

        Returns:
            (passed, reason) tuple.
        """
        if not isinstance(result, dict):
            return False, "Result is not a dict"

        if "success" not in result:
            return False, "Missing 'success' key in result"

        if not isinstance(result.get("success"), bool):
            return False, "'success' must be a boolean"

        if result["success"]:
            # Check expected success fields
            for field_name in getattr(blueprint, "return_fields_success", []):
                if field_name not in result:
                    return False, f"Missing success field: {field_name}"
        else:
            # Failure is acceptable — tool handled the error gracefully
            if "error" not in result:
                return False, "Failure result missing 'error' key"

        # Check that result/error isn't a raw exception object
        for key in ("result", "error"):
            val = result.get(key)
            if val is not None and not isinstance(val, (str, int, float, bool, list, dict)):
                return False, f"Field '{key}' is not JSON-serializable (type: {type(val).__name__})"

        return True, "Quality OK"

    @staticmethod
    def generate_dummy_params(param_schema: dict[str, str]) -> dict[str, Any]:
        """Generate safe dummy parameter values from a type schema.

        Maps Python type annotation strings to safe test values.

        Args:
            param_schema: Dict mapping parameter names to type annotation strings.

        Returns:
            Dict with dummy values for each parameter.
        """
        type_defaults: dict[str, Any] = {
            "str": "test",
            "int": 1,
            "float": 1.0,
            "bool": True,
            "list": [],
            "dict": {},
            "Any": "test",
            "Optional[str]": "test",
            "Optional[int]": 1,
            "Optional[float]": 1.0,
            "Optional[bool]": True,
            "list[str]": ["test"],
            "list[int]": [1],
            "list[float]": [1.0],
            "dict[str, str]": {"key": "value"},
            "dict[str, Any]": {"key": "value"},
        }
        result: dict[str, Any] = {}
        for name, type_str in param_schema.items():
            # Normalize type string for matching
            clean = type_str.strip().replace(" ", "")
            result[name] = type_defaults.get(clean, "test")
        return result
