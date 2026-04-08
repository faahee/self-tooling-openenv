"""Code Validator -- AST-based static analysis with 4-level import risk scoring."""
from __future__ import annotations

import ast
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# -- Default risk-level module lists ------------------------------------------------
# Overridable via config.yaml -> validator.risk_levels

_DEFAULT_LEVEL_0: frozenset[str] = frozenset({
    # Safe stdlib -- always allowed
    "json", "os", "os.path", "pathlib", "re", "datetime", "sqlite3",
    "typing", "math", "random", "string", "time", "collections",
    "itertools", "functools", "hashlib", "base64", "uuid", "csv",
    "io", "copy", "textwrap", "unicodedata", "struct", "enum",
    "dataclasses", "operator", "statistics", "decimal", "fractions",
    "calendar", "zlib", "difflib", "pprint", "urllib", "urllib.parse",
})

_DEFAULT_LEVEL_1: frozenset[str] = frozenset({
    # Data / utility libraries -- allowed with logging
    "requests", "pdfplumber", "PyPDF2", "PIL", "Pillow", "bs4",
    "openpyxl", "pandas", "numpy", "psutil", "pydantic",
    "sentence_transformers", "sklearn", "cv2", "docx", "xlrd",
    "pdfminer", "pypdfium2", "markdown", "yaml", "toml",
    "chardet", "ftfy", "tqdm", "rich", "loguru", "lxml",
    "GPUtil", "pyperclip", "speedtest", "pynvml",
})

_DEFAULT_LEVEL_2: frozenset[str] = frozenset({
    # System access -- requires human approval
    "win32api", "win32con", "winreg", "ctypes", "pyautogui",
    "keyboard", "mouse", "pygetwindow", "comtypes",
    "pynput", "pywin32", "wmi", "screeninfo",
})

_DEFAULT_LEVEL_3: frozenset[str] = frozenset({
    # Dangerous -- always blocked
    "subprocess", "shutil", "socket", "pickle", "marshal",
    "importlib", "multiprocessing", "threading",
    "sys", "builtins", "code", "codeop", "runpy",
    "signal", "shelve", "win32com",
})

# Dangerous call patterns -- always blocked regardless of import level
_DANGEROUS_CALLS: frozenset[str] = frozenset({
    "os.system",
    "subprocess.call", "subprocess.run", "subprocess.Popen",
    "subprocess.check_output", "subprocess.check_call",
    "shutil.rmtree", "shutil.rmdir", "shutil.move",
    "os.remove", "os.unlink", "os.rmdir",
    "os.makedirs", "os.mkdir",
    "sys.exit", "os._exit",
})

# Dangerous builtins -- always blocked
_DANGEROUS_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "__import__", "compile",
    "globals", "locals", "breakpoint", "exit", "quit",
})

# Blocked attribute calls on file-like objects
_BLOCKED_ATTR_WRITES: frozenset[str] = frozenset({
    "write", "write_text", "write_bytes", "unlink", "rmdir",
    "rename", "replace", "mkdir", "makedirs",
    "symlink_to", "hardlink_to", "touch",
})


@dataclass
class ValidationResult:
    """Result of code validation with risk-level scoring."""

    is_valid: bool
    risk_level: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    imports_found: list[str] = field(default_factory=list)
    imports_blocked: list[str] = field(default_factory=list)
    imports_approved: list[str] = field(default_factory=list)
    patterns_blocked: list[str] = field(default_factory=list)
    report: str = ""

    @property
    def passed(self) -> bool:
        """Alias for is_valid -- matches the JSON result schema."""
        return self.is_valid


class CodeValidator:
    """AST-based code validator with 4-level import risk scoring.

    Risk levels:
        0 -- Always allow (safe stdlib)
        1 -- Allow with logging (data/utility libraries)
        2 -- Allow with human approval (system access)
        3 -- Always block (dangerous, no exceptions)

    Unknown imports default to Level 1 (allow with logging).
    Risk levels are configurable via config.yaml -> validator.risk_levels.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the code validator.

        Args:
            config: Optional JARVIS config dict.  If it contains a
                    ``validator.risk_levels`` section, those lists override
                    the built-in defaults.
        """
        # Testing mode — allow ALL code without blocking
        self._testing_mode = False
        if config and "validator" in config:
            self._testing_mode = config["validator"].get("testing_mode", False)
            rl = config["validator"].get("risk_levels", {})
            self._level_0 = frozenset(rl.get("level_0", _DEFAULT_LEVEL_0))
            self._level_1 = frozenset(rl.get("level_1", _DEFAULT_LEVEL_1))
            self._level_2 = frozenset(rl.get("level_2", _DEFAULT_LEVEL_2))
            self._level_3 = frozenset(rl.get("level_3", _DEFAULT_LEVEL_3))
        else:
            self._level_0 = _DEFAULT_LEVEL_0
            self._level_1 = _DEFAULT_LEVEL_1
            self._level_2 = _DEFAULT_LEVEL_2
            self._level_3 = _DEFAULT_LEVEL_3

        # Precompute root-module lookups for fast classification
        self._level_0_roots = frozenset(m.split(".")[0] for m in self._level_0)
        self._level_1_roots = frozenset(m.split(".")[0] for m in self._level_1)
        self._level_2_roots = frozenset(m.split(".")[0] for m in self._level_2)
        self._level_3_roots = frozenset(m.split(".")[0] for m in self._level_3)

    # -- Public API -----------------------------------------------------------

    def validate(self, code: str) -> ValidationResult:
        """Synchronous validation with risk scoring.

        Level 2 imports are treated as BLOCKED when called synchronously
        (no UI available for approval).  Use ``validate_async()`` for
        interactive Level 2 approval.

        Args:
            code: Python source code string.

        Returns:
            ValidationResult with is_valid flag, risk_level, errors, etc.
        """
        if self._testing_mode:
            return ValidationResult(is_valid=True)
        return self._run_validation(code, level2_approved=set())

    async def validate_async(
        self,
        code: str,
        ui_layer: Any | None = None,
        interaction_logger: Any | None = None,
    ) -> ValidationResult:
        """Async validation with Level 2 human-approval support.

        Args:
            code: Python source code string.
            ui_layer: UILayer instance for Level 2 approval prompts.
            interaction_logger: InteractionLogger for Level 1+ event logging.

        Returns:
            ValidationResult with full risk report.
        """
        if self._testing_mode:
            return ValidationResult(is_valid=True)
        # Preliminary pass -- discover Level 2 imports without blocking
        prelim = self._run_validation(
            code, level2_approved=set(), dry_run_level2=True,
        )

        # If syntax error or no Level 2 imports, return early
        level2_pending = [
            imp for imp in prelim.imports_found
            if self._classify_import(imp) == 2
        ]

        if not level2_pending:
            self._log_imports(interaction_logger, prelim)
            return prelim

        # Prompt user for each Level 2 import
        approved_level2: set[str] = set()
        if ui_layer:
            for imp in level2_pending:
                approved = await self._prompt_approval(ui_layer, imp)
                if approved:
                    approved_level2.add(imp)

        # Re-run with approvals applied
        result = self._run_validation(code, level2_approved=approved_level2)
        self._log_imports(interaction_logger, result)
        return result

    # -- Core validation pipeline -------------------------------------------------

    def _run_validation(
        self,
        code: str,
        level2_approved: set[str],
        dry_run_level2: bool = False,
    ) -> ValidationResult:
        """Run the full 6-step validation pipeline.

        Args:
            code: Python source code.
            level2_approved: Set of Level 2 modules the user approved.
            dry_run_level2: If True, skip blocking Level 2 (preliminary scan).

        Returns:
            ValidationResult with all findings.
        """
        errors: list[str] = []
        warnings: list[str] = []
        imports_found: list[str] = []
        imports_blocked: list[str] = []
        imports_approved: list[str] = []
        patterns_blocked: list[str] = []
        max_risk = 0

        # Step 1: Syntax check (ast.parse)
        syntax_errors = self._check_syntax(code)
        if syntax_errors:
            return ValidationResult(
                is_valid=False, risk_level=0, errors=syntax_errors,
                report="Syntax error -- cannot analyze further.",
            )

        tree = ast.parse(code)

        # Step 2: Extract and classify all imports
        imp_errors, imp_data = self._check_imports(
            tree, level2_approved, dry_run_level2,
        )
        imports_found = imp_data["found"]
        imports_blocked = imp_data["blocked"]
        imports_approved = list(level2_approved & set(imports_found))
        errors.extend(imp_errors)
        max_risk = max(max_risk, imp_data["max_risk"])

        for imp in imp_data["level1"]:
            warnings.append(f"Level 1 import (logged): {imp}")

        # Step 3-4: Dangerous call patterns + blocked builtins
        call_errors = self._check_dangerous_calls(tree)
        for e in call_errors:
            patterns_blocked.append(e)
            errors.append(e)
        if call_errors:
            max_risk = max(max_risk, 3)

        builtin_errors = self._check_blocked_builtins(tree)
        for e in builtin_errors:
            patterns_blocked.append(e)
            errors.append(e)
        if builtin_errors:
            max_risk = max(max_risk, 3)

        # Step 5: Dangerous code patterns (open-write, traversal, loops, etc.)
        pattern_errors = self._check_dangerous_patterns(tree)
        for e in pattern_errors:
            patterns_blocked.append(e)
            errors.append(e)
        if pattern_errors:
            max_risk = max(max_risk, 3)

        # Step 6: Function structure
        struct_errors = self._check_function_structure(tree)
        errors.extend(struct_errors)

        # Line count
        line_count = len(code.strip().splitlines())
        if line_count > 60:
            errors.append(
                f"Code has {line_count} lines (max allowed: 60, recommended: 50)"
            )
        elif line_count > 50:
            warnings.append(f"Code has {line_count} lines (recommended max: 50)")

        # Build human-readable report
        report_lines: list[str] = []
        if imports_found:
            report_lines.append(f"Imports: {', '.join(imports_found)}")
        if imports_blocked:
            report_lines.append(f"Blocked: {', '.join(imports_blocked)}")
        if imports_approved:
            report_lines.append(f"Approved (Level 2): {', '.join(imports_approved)}")
        if patterns_blocked:
            report_lines.append(f"Dangerous patterns: {'; '.join(patterns_blocked)}")
        if not errors:
            report_lines.append(f"PASSED -- max risk level: {max_risk}")
        else:
            report_lines.append(f"FAILED -- {len(errors)} error(s)")
        report = "\n".join(report_lines)

        return ValidationResult(
            is_valid=len(errors) == 0,
            risk_level=max_risk,
            errors=errors,
            warnings=warnings,
            imports_found=imports_found,
            imports_blocked=imports_blocked,
            imports_approved=imports_approved,
            patterns_blocked=patterns_blocked,
            report=report,
        )

    # -- Step 1: Syntax -------------------------------------------------------

    def _check_syntax(self, code: str) -> list[str]:
        """Verify code is syntactically valid Python.

        Args:
            code: Python source code.

        Returns:
            List of syntax error messages.
        """
        try:
            ast.parse(code)
            return []
        except SyntaxError as exc:
            return [f"Syntax error at line {exc.lineno}: {exc.msg}"]

    # -- Step 2: Import classification ----------------------------------------

    def _classify_import(self, module_name: str) -> int:
        """Classify a module name to a risk level (0-3).

        Checks the full dotted name first, then the root package.
        Unknown modules default to Level 1 (allow with logging).

        Args:
            module_name: Dotted module name (e.g. ``os.path``).

        Returns:
            Risk level integer 0-3.
        """
        root = module_name.split(".")[0]

        # Check most dangerous first for fast rejection
        if module_name in self._level_3 or root in self._level_3_roots:
            return 3
        if module_name in self._level_2 or root in self._level_2_roots:
            return 2
        if module_name in self._level_1 or root in self._level_1_roots:
            return 1
        if module_name in self._level_0 or root in self._level_0_roots:
            return 0

        # Unknown module -> Level 1 (allow with logging, not silently)
        return 1

    def _check_imports(
        self,
        tree: ast.Module,
        level2_approved: set[str],
        dry_run_level2: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        """Walk AST, extract all imports, and score each by risk level.

        Args:
            tree: Parsed AST module.
            level2_approved: Set of Level 2 module names the user approved.
            dry_run_level2: If True, skip erroring on Level 2 (prelim scan).

        Returns:
            Tuple of (error_list, data_dict).  data_dict keys:
            found, blocked, level1, max_risk.
        """
        errors: list[str] = []
        found: list[str] = []
        blocked: list[str] = []
        level1_imports: list[str] = []
        max_risk = 0

        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]

            for mod in modules:
                if mod not in found:
                    found.append(mod)

                risk = self._classify_import(mod)
                max_risk = max(max_risk, risk)

                if risk == 3:
                    errors.append(f"Blocked import (Level 3): {mod}")
                    if mod not in blocked:
                        blocked.append(mod)
                elif risk == 2:
                    root = mod.split(".")[0]
                    if root in level2_approved or mod in level2_approved:
                        pass  # user approved
                    elif dry_run_level2:
                        pass  # preliminary scan -- do not error
                    else:
                        errors.append(
                            f"Blocked import (Level 2 -- requires approval): {mod}"
                        )
                        if mod not in blocked:
                            blocked.append(mod)
                elif risk == 1:
                    if mod not in level1_imports:
                        level1_imports.append(mod)

        return errors, {
            "found": found,
            "blocked": blocked,
            "level1": level1_imports,
            "max_risk": max_risk,
        }

    # -- Step 3: Dangerous call patterns --------------------------------------

    def _check_dangerous_calls(self, tree: ast.Module) -> list[str]:
        """Check for dangerous function/method calls via AST walk.

        Args:
            tree: Parsed AST module.

        Returns:
            List of violation messages.
        """
        errors: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and call_name in _DANGEROUS_CALLS:
                    errors.append(f"Blocked dangerous call: {call_name}()")
        return errors

    # -- Step 4: Blocked builtins ---------------------------------------------

    def _check_blocked_builtins(self, tree: ast.Module) -> list[str]:
        """Check for use of blocked builtin functions.

        Args:
            tree: Parsed AST module.

        Returns:
            List of violation messages.
        """
        errors: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in _DANGEROUS_BUILTINS
                ):
                    errors.append(f"Blocked builtin call: {node.func.id}()")
        return errors

    # -- Step 5: Dangerous code patterns --------------------------------------

    def _check_dangerous_patterns(self, tree: ast.Module) -> list[str]:
        """Scan AST for dangerous code patterns.

        Checks for:
        - open() in write mode (except brain/tools/generated/)
        - Path traversal (../ or absolute paths outside JARVIS/)
        - Infinite loops (while True without break)
        - Blocked attribute write calls (.write(), .unlink(), etc.)

        Args:
            tree: Parsed AST module.

        Returns:
            List of violation messages.
        """
        errors: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # open() in write/append mode
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    mode = self._extract_open_mode(node)
                    if any(c in mode for c in ("w", "a", "x")):
                        # Allow writes ONLY to brain/tools/generated/
                        path_arg = self._extract_first_string_arg(node)
                        if (
                            not path_arg
                            or "brain/tools/generated" not in path_arg.replace("\\", "/")
                        ):
                            errors.append(
                                "Blocked: open() in write/append mode "
                                "(only brain/tools/generated/ allowed)"
                            )

                # Blocked attribute writes (.write(), .unlink(), etc.)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in _BLOCKED_ATTR_WRITES:
                        errors.append(
                            f"Blocked: .{node.func.attr}() is not allowed in generated tools"
                        )

            # Path traversal in string constants
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                if "../" in val or "..\\" in val:
                    errors.append(
                        f"Blocked: path traversal detected ('{val[:60]}')"
                    )
                # Absolute path outside JARVIS workspace
                if (
                    len(val) >= 3
                    and val[0].isalpha()
                    and val[1] == ":"
                    and val[2] in ("/", "\\")
                    and "jarvis" not in val.lower()
                ):
                    errors.append(
                        f"Blocked: absolute path outside JARVIS ('{val[:60]}')"
                    )

            # while True without break
            if isinstance(node, ast.While):
                if self._is_while_true(node) and not self._has_break(node):
                    errors.append(
                        "Blocked: while True loop without break statement"
                    )

        return errors

    # -- Step 6: Function structure -------------------------------------------

    def _check_function_structure(self, tree: ast.Module) -> list[str]:
        """Verify exactly one top-level function definition, no classes.

        Args:
            tree: Parsed AST module.

        Returns:
            List of structural violation messages.
        """
        errors: list[str] = []

        functions = [
            n for n in ast.iter_child_nodes(tree)
            if isinstance(n, ast.FunctionDef)
        ]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        if classes:
            errors.append("Class definitions are not allowed in generated tools")
        if not functions:
            errors.append(
                "No function definition found -- tool must define exactly one function"
            )
        elif len(functions) > 1:
            errors.append(
                f"Found {len(functions)} top-level functions -- tool must have exactly one"
            )

        return errors

    # -- AST helper methods ---------------------------------------------------

    @staticmethod
    def _get_call_name(node: ast.Call) -> str | None:
        """Extract the dotted name of a function call from an AST Call node.

        Examples:
            os.system(...)  -> "os.system"
            eval(...)       -> "eval"
            a.b.c(...)      -> "a.b.c"

        Args:
            node: An ast.Call node.

        Returns:
            Dotted name string, or None if the call cannot be resolved.
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None

    @staticmethod
    def _extract_open_mode(node: ast.Call) -> str:
        """Extract the file mode from an open() call.

        Checks positional arg #2 and the ``mode`` keyword argument.

        Args:
            node: An ast.Call node for an open() call.

        Returns:
            Mode string (defaults to ``"r"`` if not found).
        """
        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        return mode if isinstance(mode, str) else "r"

    @staticmethod
    def _extract_first_string_arg(node: ast.Call) -> str | None:
        """Extract the first string constant argument from a Call node.

        Args:
            node: An ast.Call node.

        Returns:
            The first string constant argument, or None.
        """
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        return None

    @staticmethod
    def _is_while_true(node: ast.While) -> bool:
        """Check if a While node has a constant True test."""
        return isinstance(node.test, ast.Constant) and node.test.value is True

    @staticmethod
    def _has_break(node: ast.While) -> bool:
        """Check if a While loop body contains a break statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Break):
                return True
        return False

    # -- Level 2 interactive approval -----------------------------------------

    @staticmethod
    async def _prompt_approval(ui_layer: Any, module_name: str) -> bool:
        """Prompt the user via UILayer for Level 2 import approval.

        Timeout: 30 seconds -> auto-reject.

        Args:
            ui_layer: UILayer instance with prompt_import_approval method.
            module_name: The module requesting approval.

        Returns:
            True if the user approves, False on rejection or timeout.
        """
        try:
            approved = await asyncio.wait_for(
                ui_layer.prompt_import_approval(module_name),
                timeout=30.0,
            )
            return bool(approved)
        except asyncio.TimeoutError:
            logger.warning(
                "Level 2 approval timed out for '%s' -- auto-rejected", module_name
            )
            if hasattr(ui_layer, "console"):
                ui_layer.console.print(
                    f"  [dim yellow]>> Import '{module_name}' auto-rejected "
                    f"(30s timeout)[/dim yellow]"
                )
            return False
        except Exception as exc:
            logger.warning(
                "Level 2 approval failed for '%s': %s", module_name, exc
            )
            return False

    # -- Import event logging -------------------------------------------------

    @staticmethod
    def _log_imports(interaction_logger: Any, result: "ValidationResult") -> None:
        """Log Level 1+ import events to the interaction logger.

        Args:
            interaction_logger: InteractionLogger with log_tool_event method.
            result: The completed ValidationResult.
        """
        try:
            if not hasattr(interaction_logger, "log_tool_event"):
                return

            details: dict[str, Any] = {
                "risk_level": result.risk_level,
                "imports_found": result.imports_found,
                "blocked": result.imports_blocked,
                "approved": result.imports_approved,
            }
            if result.risk_level >= 1:
                interaction_logger.log_tool_event(
                    "import_risk_assessment", "code_validator", details
                )
        except Exception as exc:
            logger.debug("Failed to log import event: %s", exc)
