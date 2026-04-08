"""Tool Synthesizer — 5-phase reasoning pipeline for Python tool code generation.

Phase 1: Deep Reasoning   → ToolBlueprint dataclass
Phase 2: Blueprint Valid.  → validates completeness before any code
Phase 3: Code Generation   → translates blueprint to Python
Phase 4: Self Review       → LLM reviews own code against checklist
Phase 5: Sandbox + Quality → executes and checks result quality
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.llm_core import TaskType

logger = logging.getLogger(__name__)


# ── ToolBlueprint dataclass ───────────────────────────────────────────────────

@dataclass
class ToolBlueprint:
    """Complete specification produced by Phase 1 reasoning."""

    tool_name: str
    task_description: str
    approach: str = ""
    imports: list[dict] = field(default_factory=list)
    function_signature: str = ""
    parameters: list[dict] = field(default_factory=list)
    return_fields_success: list[str] = field(default_factory=list)
    return_fields_failure: list[str] = field(default_factory=list)
    pseudocode: list[str] = field(default_factory=list)
    edge_cases: list[dict] = field(default_factory=list)
    windows_considerations: list[str] = field(default_factory=list)
    risk_level: int = 0
    estimated_lines: int = 30
    confidence: float = 0.5


# ── Prompts ───────────────────────────────────────────────────────────────────

_REASONING_SYSTEM_PROMPT: str = (
    "You are an expert Python software architect on Windows 11.\n"
    "Analyze the task and answer EVERY question below. Output ONLY the structured answers.\n"
    "Do NOT write any code. This is planning only.\n\n"

    "QUESTION 1 — WHAT EXACTLY DOES THIS TOOL NEED TO DO?\n"
    "  TASK_SUMMARY: <one sentence>\n"
    "  INPUT_PARAMS: <param_name (type) = default — description> (one per line)\n"
    "  HAPPY_PATH: <what happens on success>\n"
    "  FAILURE_CASES: <each way this can fail, one per line>\n"
    "  SUCCESS_LOOKS_LIKE: <exact success return value description>\n"
    "  FAILURE_LOOKS_LIKE: <exact error return value description>\n\n"

    "QUESTION 2 — WHAT IS THE BEST TECHNICAL APPROACH?\n"
    "  Available stdlib: os, sys, pathlib, subprocess, socket, json, csv, re, datetime,\n"
    "    threading, asyncio, urllib, http, smtplib, imaplib, zipfile, hashlib, base64,\n"
    "    uuid, math, random, statistics, collections, time, io, struct, tempfile\n"
    "  Available third-party: requests, httpx, psutil, pyautogui, pygetwindow, pycaw,\n"
    "    pdfplumber, python-docx, pandas, numpy, PIL, cv2, beautifulsoup4, feedparser,\n"
    "    pyperclip, pyttsx3, win32api, win32con, wmi, GPUtil, pynvml, speedtest-cli,\n"
    "    googletrans, langdetect, qrcode, pywin32, comtypes, pywinauto\n"
    "  CHOSEN_APPROACH: <exact library and function to use, with reason>\n"
    "  WHY: <why this is the most reliable approach for Windows 11>\n\n"

    "QUESTION 3 — EXACT IMPORTS NEEDED?\n"
    "  For each import:\n"
    "  IMPORT: <statement>\n"
    "  WHY: <what specific function is used from it>\n"
    "  RISK: <0=safe stdlib, 1=safe third-party, 2=system access, 3=dangerous>\n"
    "  ALTERNATIVE: <fallback if this import fails>\n\n"

    "QUESTION 4 — EXACT FUNCTION SIGNATURE?\n"
    "  FUNCTION_NAME: <snake_case name>\n"
    "  SIGNATURE: <complete def line with type hints and -> dict>\n"
    "  DOCSTRING: <one sentence describing what it does>\n\n"

    "QUESTION 5 — STEP BY STEP LOGIC?\n"
    "  Write pseudocode:\n"
    "  STEP 1: <what to do> using <specific function call>\n"
    "  STEP 2: ...\n"
    "  (at least 4 steps)\n\n"

    "QUESTION 6 — WINDOWS 11 CONSIDERATIONS?\n"
    "  List any Windows-specific concerns (paths, admin, registry, COM, UAC).\n"
    "  CONSIDERATION: <text>\n\n"

    "QUESTION 7 — EXACT RETURN STRUCTURE?\n"
    "  SUCCESS_FIELDS: success, result, data, summary, source, executed_at\n"
    "  FAILURE_FIELDS: success, error, error_type, attempted, suggestion, executed_at\n"
    "  (Confirm or modify these fields for this specific tool)\n\n"

    "QUESTION 8 — EDGE CASES?\n"
    "  EDGE_CASE: <what can go wrong> → HANDLER: <how to handle it>\n"
    "  (list every edge case)\n\n"

    "ADDITIONAL:\n"
    "  RISK_LEVEL: <0-3 overall>\n"
    "  ESTIMATED_LINES: <expected code length>\n"
    "  CONFIDENCE: <0.0-1.0 how confident in approach>\n"
)

_CODE_GEN_SYSTEM_PROMPT: str = (
    "You are translating a blueprint into Python code.\n"
    "Follow the blueprint EXACTLY. Do not deviate.\n"
    "Do not add features not in the blueprint.\n"
    "Do not use libraries not in the blueprint imports list.\n\n"
    "RULES:\n"
    "1. ALL imports go INSIDE the function body — never at module level\n"
    "2. Function must be completely self-contained\n"
    "3. Every code path must return a dict with keys: success (bool), result (str), error (str or None)\n"
    "   - On SUCCESS: success=True, result=<human-readable string>, error=None\n"
    "   - On FAILURE: success=False, result=None, error=<descriptive error string, NEVER None>\n"
    "4. Never return None, never raise exceptions to caller\n"
    "5. Add inline comments explaining each major step\n"
    "6. Follow the pseudocode steps in exact order\n"
    "7. Maximum 60 lines of code\n"
    "8. Include try/except around main logic\n"
    "9. Handle ImportError for each third-party library\n"
    "10. Only import libraries you actually use\n\n"
    "EXAMPLE of correct structure:\n"
    "```python\n"
    "def example_tool(param: str = 'default') -> dict:\n"
    '    """Does something useful."""\n'
    "    try:\n"
    "        import some_module\n"
    "        # do the work\n"
    "        value = some_module.do_thing(param)\n"
    '        return {"success": True, "result": f"Got: {value}", "error": None}\n'
    "    except ImportError:\n"
    '        return {"success": False, "result": None, "error": "some_module not installed"}\n'
    "    except Exception as e:\n"
    '        return {"success": False, "result": None, "error": str(e)}\n'
    "```\n\n"
    "COMMON PATTERNS (use these exact patterns when applicable):\n"
    "- Uptime/boot time: boot = datetime.datetime.fromtimestamp(psutil.boot_time()); "
    "uptime = datetime.datetime.now() - boot; "
    "hours, rem = divmod(int(uptime.total_seconds()), 3600); "
    "mins, secs = divmod(rem, 60)\n"
    "- File paths: always use pathlib.Path, never raw strings\n"
    "- Recursive file scan with permission handling:\n"
    "  files = []\n"
    "  for root, dirs, names in os.walk('C:/'):\n"
    "      dirs[:] = [d for d in dirs if d not in ('$Recycle.Bin','System Volume Information','Windows')]\n"
    "      for name in names:\n"
    "          fp = os.path.join(root, name)\n"
    "          try: files.append((fp, os.path.getsize(fp)))\n"
    "          except (PermissionError, OSError): pass\n"
    "- HTTP requests: response = requests.get(url, timeout=10); response.raise_for_status()\n"
    "- Subprocess: subprocess.run([...], capture_output=True, text=True, timeout=15)\n"
    "- WiFi signal strength on Windows:\n"
    "  r = subprocess.run(['netsh','wlan','show','interfaces'], capture_output=True, text=True, timeout=10)\n"
    "  for line in r.stdout.splitlines():\n"
    "      if 'Signal' in line: signal = line.split(':')[1].strip()\n"
    "      if 'SSID' in line and 'BSSID' not in line: ssid = line.split(':')[1].strip()\n"
    "  (No parameters needed — reads current connection automatically)\n\n"
    "Output ONLY the Python code in a ```python block. No explanations.\n"
)

_SELF_REVIEW_SYSTEM_PROMPT: str = (
    "Review this Python code critically before it is executed.\n"
    "Check every item in this list and report issues:\n\n"
    "REVIEW CHECKLIST:\n"
    "[ ] Are ALL imports inside the function body? (not at module level)\n"
    "[ ] Does the function return a dict in ALL code paths?\n"
    "[ ] Does success return include: success, result, error?\n"
    "[ ] Is there a try/except around the main logic?\n"
    "[ ] Are all parameters validated at the start?\n"
    "[ ] Does it handle ImportError for each third party library?\n"
    "[ ] Does it use pathlib for all Windows paths?\n"
    "[ ] Are there any syntax errors or obvious bugs?\n"
    "[ ] Does the code actually accomplish the task in the blueprint?\n"
    "[ ] Are there any hardcoded values that should be parameters?\n"
    "[ ] Is the result field a human-readable string (not raw bytes or object)?\n\n"
    "For each issue found:\n"
    "ISSUE: <what is wrong>\n"
    "FIX: <exact fix>\n"
    "SEVERITY: CRITICAL / WARNING / MINOR\n\n"
    "If no CRITICAL issues: respond with APPROVED on its own line.\n"
    "If CRITICAL issues: respond with REJECTED on its own line, then list all fixes.\n"
    "Then output the FIXED code in a ```python block.\n"
)


class ToolSynthesizer:
    """5-phase tool synthesis pipeline: Reason → Validate → Generate → Review → Sandbox."""

    def __init__(
        self,
        config: dict,
        llm_core,
        code_validator,
        sandbox_executor,
        tool_registry,
        ui_layer,
    ) -> None:
        """Initialize the tool synthesizer.

        Args:
            config: Full JARVIS configuration dictionary.
            llm_core: LLMCore instance for Ollama calls.
            code_validator: CodeValidator instance.
            sandbox_executor: SandboxExecutor instance.
            tool_registry: ToolRegistry instance.
            ui_layer: UILayer instance for progress display.
        """
        self.config = config
        self.llm = llm_core
        self.validator = code_validator
        self.sandbox = sandbox_executor
        self.registry = tool_registry
        self.ui = ui_layer
        self.tools_dir = Path(config["brain"]["tools_dir"])
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries: int = config["brain"]["max_synthesis_retries"]
        self.mcp_client = None  # Attached by main.py if MCP enabled

    # ══════════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    async def synthesize(
        self,
        task_description: str,
        existing_tool_name: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run the complete 5-phase synthesis pipeline.

        Args:
            task_description: What the user wants the tool to do.
            existing_tool_name: Name of existing tool to upgrade (if improving).
            plan: Optional pre-existing plan dict (from decision engine reasoning).

        Returns:
            Tool metadata dict on success, None after all retries exhausted.
        """
        # ── MCP shortcut ─────────────────────────────────────────────────
        if self.mcp_client and self.mcp_client.enabled:
            match = self.mcp_client.find_tool(task_description)
            if match and match.get("match_score", 0) >= 0.5:
                logger.info(
                    "MCP tool found: %s (score=%.2f) — skipping synthesis.",
                    match["name"], match["match_score"],
                )
                self.ui.console.print(
                    f"  [dim green]>> MCP tool available: {match['name']} "
                    f"(score={match['match_score']:.2f}) — no synthesis needed[/dim green]"
                )
                return {
                    "name": match["name"],
                    "description": match.get("description", ""),
                    "source": "mcp",
                    "server": match.get("server", ""),
                    "mcp_tool": True,
                }

        # Load existing code if upgrading
        existing_code: str | None = None
        if existing_tool_name:
            tool = self.registry.get_by_name(existing_tool_name)
            if tool:
                code_path = Path(tool["code_path"])
                if code_path.exists():
                    existing_code = code_path.read_text(encoding="utf-8")

        # Determine display name
        display_name = existing_tool_name
        if not display_name and plan and plan.get("tool_name"):
            display_name = plan["tool_name"]
        if not display_name:
            display_name = "new_tool"
        self.ui.show_synthesis_header(display_name)

        # ── PHASE 1: Deep Reasoning → ToolBlueprint ─────────────────────
        blueprint = await self._phase1_reasoning(task_description, existing_code)
        if blueprint is None:
            self.ui.show_synthesis_failure("Phase 1 reasoning failed — could not create blueprint")
            return None

        # ── PHASE 2: Blueprint Validation ────────────────────────────────
        valid, reason = self._phase2_validate_blueprint(blueprint)
        if not valid:
            self.ui.show_synthesis_failure(f"Phase 2 validation failed: {reason}")
            return None

        # ── PHASE 3: Code Generation ──────────────────────────────────────
        code = await self._phase3_generate_code(blueprint, existing_code)
        if code is None:
            self.ui.show_synthesis_failure("Phase 3 code generation failed")
            return None

        # ── PHASE 4: Self Review ──────────────────────────────────────────
        code = await self._phase4_self_review(code, blueprint)
        if code is None:
            self.ui.show_synthesis_failure("Phase 4 self-review failed after retries")
            return None

        # ── Validation via CodeValidator ─────────────────────────────────
        validation = await self.validator.validate_async(code, ui_layer=self.ui)
        if not validation.is_valid:
            logger.warning("Code validation failed: %s", "; ".join(validation.errors))
            self.ui.show_synthesis_failure(f"Code validation: {'; '.join(validation.errors)}")
            return None

        # ── PHASE 5: Sandbox + Quality Check ─────────────────────────────
        metadata = self._extract_metadata(code)
        if not metadata:
            self.ui.show_synthesis_failure("Could not extract function metadata from code")
            return None

        function_name = metadata["name"]
        param_schema = metadata["parameters"]

        sandbox_ok, sandbox_result = await self._phase5_sandbox(
            code, function_name, param_schema, blueprint,
        )
        if not sandbox_ok:
            self.ui.show_synthesis_failure(
                f"Sandbox failed: {sandbox_result.get('error', 'unknown')}"
            )
            return None

        # ── Register ─────────────────────────────────────────────────────
        effective_description = blueprint.task_description
        tool_name = existing_tool_name or blueprint.tool_name
        version = (
            self.registry.get_version(tool_name) + 1
            if existing_tool_name
            else 1
        )
        code_path_obj = self.tools_dir / f"{tool_name}_v{version}.py"
        code_path_obj.write_text(code, encoding="utf-8")

        tool_id = self.registry.register(
            name=tool_name,
            description=effective_description,
            code_path=str(code_path_obj),
            parameters=param_schema,
            created_by="agent",
            version=version,
            tags=self._extract_tags(effective_description),
        )

        self.ui.show_phase_status(
            5, "REGISTERED",
            f"{tool_name}_v{version}.py → brain.db",
        )

        return {
            "tool_id": tool_id,
            "name": tool_name,
            "description": effective_description,
            "code_path": str(code_path_obj),
            "parameters": param_schema,
            "version": version,
            "function_name": function_name,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 — DEEP REASONING
    # ══════════════════════════════════════════════════════════════════════════

    async def _phase1_reasoning(
        self,
        task: str,
        existing_code: str | None = None,
        feedback: str | None = None,
    ) -> ToolBlueprint | None:
        """Run deep reasoning to produce a ToolBlueprint.

        Args:
            task: Natural language task description.
            existing_code: Existing code to improve (if upgrading).
            feedback: Feedback from a failed blueprint validation attempt.

        Returns:
            ToolBlueprint or None on failure.
        """
        self.ui.show_phase_status(1, "RUNNING", "Deep reasoning about the task...")

        prompt = f'Task: "{task}"\n\nAnswer every question in the structured format.'
        if existing_code:
            prompt += f"\n\nExisting code to improve:\n```python\n{existing_code}\n```"
        if feedback:
            prompt += (
                f"\n\nPREVIOUS BLUEPRINT WAS REJECTED: {feedback}\n"
                f"Fix the issues and provide a complete blueprint."
            )

        response = await self.llm.generate(
            prompt,
            task_type=TaskType.TOOL_BLUEPRINT,
            system_prompt=_REASONING_SYSTEM_PROMPT,
        )

        blueprint = self._parse_blueprint(response, task)
        if blueprint:
            self.ui.show_blueprint(blueprint)
        else:
            logger.warning("Failed to parse blueprint from LLM response")

        return blueprint

    def _parse_blueprint(self, response: str, task: str) -> ToolBlueprint | None:
        """Parse LLM reasoning response into a ToolBlueprint.

        Args:
            response: Raw LLM reasoning output.
            task: Original task description.

        Returns:
            ToolBlueprint or None on parse failure.
        """
        try:
            bp = ToolBlueprint(tool_name="", task_description=task)

            # ── Tool name ─────────────────────────────────────────────
            m = re.search(r"FUNCTION_NAME:\s*(\w+)", response)
            if m:
                bp.tool_name = m.group(1).strip()
            else:
                # Fallback: derive from task
                words = re.findall(r"[a-zA-Z]+", task.lower())
                bp.tool_name = "_".join(words[:4]) + "_tool" if words else "custom_tool"

            # ── Approach ──────────────────────────────────────────────
            m = re.search(r"CHOSEN_APPROACH:\s*(.+?)(?:\n|$)", response)
            bp.approach = m.group(1).strip() if m else ""

            # If CHOSEN_APPROACH is empty, try WHY field
            if not bp.approach or len(bp.approach) < 10:
                m = re.search(r"WHY:\s*(.+?)(?:\n|$)", response)
                if m:
                    bp.approach = m.group(1).strip()

            # ── Imports ───────────────────────────────────────────────
            imports: list[dict] = []
            import_blocks = re.findall(
                r"IMPORT:\s*(.+?)(?:\n)"
                r".*?WHY:\s*(.+?)(?:\n)"
                r".*?RISK:\s*(\d).*?(?:\n)"
                r".*?ALTERNATIVE:\s*(.+?)(?:\n|$)",
                response, re.DOTALL,
            )
            for stmt, why, risk, alt in import_blocks:
                imports.append({
                    "statement": stmt.strip(),
                    "why": why.strip(),
                    "risk": int(risk),
                    "alternative": alt.strip(),
                })
            # Fallback: grab any "import ..." lines
            if not imports:
                for m_imp in re.finditer(r"(?:from\s+\w+\s+)?import\s+[\w, ]+", response):
                    imports.append({
                        "statement": m_imp.group(0).strip(),
                        "why": "detected from response",
                        "risk": 1,
                        "alternative": "none",
                    })
            bp.imports = imports

            # ── Function signature ────────────────────────────────────
            m = re.search(r"SIGNATURE:\s*(def\s+\w+\(.+?\)\s*->\s*\w+)", response, re.DOTALL)
            if m:
                bp.function_signature = m.group(1).strip()
            else:
                # Try any def line
                m = re.search(r"(def\s+\w+\([^)]*\)\s*(?:->\s*\w+)?)", response)
                if m:
                    bp.function_signature = m.group(1).strip()

            # ── Parameters ────────────────────────────────────────────
            params: list[dict] = []
            param_matches = re.findall(
                r"INPUT_PARAMS?:?\s*(.+?)(?:\n|$)", response,
            )
            for pm_line in param_matches:
                for pm in re.finditer(
                    r"(\w+)\s*\((\w+)\)\s*(?:=\s*(\S+))?\s*[-—]\s*(.+)",
                    pm_line,
                ):
                    params.append({
                        "name": pm.group(1),
                        "type": pm.group(2),
                        "default": pm.group(3) or "None",
                        "description": pm.group(4).strip(),
                    })
            # Also try to parse from function signature
            if not params and bp.function_signature:
                sig_match = re.search(r"\((.+?)\)", bp.function_signature)
                if sig_match:
                    for arg in sig_match.group(1).split(","):
                        arg = arg.strip()
                        if not arg:
                            continue
                        parts = re.match(r"(\w+)\s*:\s*(\w+)(?:\s*=\s*(.+))?", arg)
                        if parts:
                            params.append({
                                "name": parts.group(1),
                                "type": parts.group(2),
                                "default": parts.group(3) or "None",
                                "description": "",
                            })
            bp.parameters = params

            # ── Pseudocode ────────────────────────────────────────────
            steps: list[str] = []
            for sm in re.finditer(r"STEP\s*\d+:\s*(.+?)(?:\n|$)", response):
                steps.append(sm.group(1).strip())
            # Fallback: numbered list after STEP header
            if not steps:
                step_section = re.search(
                    r"(?:STEP.BY.STEP|LOGIC|PSEUDOCODE)[:\s]*\n((?:\s*\d+\..+\n?)+)",
                    response, re.IGNORECASE,
                )
                if step_section:
                    steps = [
                        s.strip()
                        for s in re.findall(r"\d+\.\s*(.+)", step_section.group(1))
                    ]
            bp.pseudocode = steps

            # ── Edge cases ────────────────────────────────────────────
            edge_cases: list[dict] = []
            ec_matches = re.findall(
                r"EDGE_CASE:\s*(.+?)\s*(?:→|->|HANDLER:)\s*(.+?)(?:\n|$)",
                response,
            )
            for case, handler in ec_matches:
                edge_cases.append({"case": case.strip(), "handler": handler.strip()})
            # Fallback: bullet lines with error/fail keywords
            if not edge_cases:
                for ecm in re.finditer(
                    r"[-•]\s*(?:What if\s+)?(.+?)\s*(?:→|->|:)\s*(.+?)(?:\n|$)",
                    response,
                ):
                    if any(kw in ecm.group(1).lower() for kw in (
                        "error", "fail", "empty", "not ", "timeout",
                        "denied", "missing", "install", "import",
                    )):
                        edge_cases.append({
                            "case": ecm.group(1).strip(),
                            "handler": ecm.group(2).strip(),
                        })
            bp.edge_cases = edge_cases

            # ── Windows considerations ────────────────────────────────
            win_considerations: list[str] = []
            for wm in re.finditer(r"CONSIDERATION:\s*(.+?)(?:\n|$)", response):
                win_considerations.append(wm.group(1).strip())
            bp.windows_considerations = win_considerations

            # ── Return fields ─────────────────────────────────────────
            success_fields = ["success", "result", "error"]
            failure_fields = ["success", "error", "error_type", "attempted", "suggestion"]

            m_sf = re.search(r"SUCCESS_FIELDS:\s*(.+?)(?:\n|$)", response)
            if m_sf:
                fields = [f.strip() for f in m_sf.group(1).split(",") if f.strip()]
                if fields:
                    success_fields = fields

            m_ff = re.search(r"FAILURE_FIELDS:\s*(.+?)(?:\n|$)", response)
            if m_ff:
                fields = [f.strip() for f in m_ff.group(1).split(",") if f.strip()]
                if fields:
                    failure_fields = fields

            bp.return_fields_success = success_fields
            bp.return_fields_failure = failure_fields

            # ── Risk / lines / confidence ─────────────────────────────
            m = re.search(r"RISK_LEVEL:\s*(\d)", response)
            bp.risk_level = int(m.group(1)) if m else max(
                (imp.get("risk", 0) for imp in bp.imports), default=0,
            )

            m = re.search(r"ESTIMATED_LINES:\s*(\d+)", response)
            bp.estimated_lines = int(m.group(1)) if m else 30

            m = re.search(r"CONFIDENCE:\s*([\d.]+)", response)
            bp.confidence = float(m.group(1)) if m else 0.5

            # Ensure tool_name is valid Python identifier
            bp.tool_name = re.sub(r"[^a-z0-9_]", "_", bp.tool_name.lower())
            if not bp.tool_name or bp.tool_name[0].isdigit():
                bp.tool_name = "tool_" + bp.tool_name

            return bp

        except Exception as exc:
            logger.warning("Blueprint parsing error: %s", exc)
            return None

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2 — BLUEPRINT VALIDATION
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2_validate_blueprint(
        self, blueprint: ToolBlueprint,
    ) -> tuple[bool, str]:
        """Validate blueprint completeness before code generation.

        Args:
            blueprint: The ToolBlueprint to validate.

        Returns:
            (is_valid, reason) tuple.
        """
        if not blueprint.approach or len(blueprint.approach) < 10:
            return False, "Approach not specific enough. Must name exact library and reason."

        if not blueprint.imports:
            return False, "No imports specified. Every tool needs at least one import."
        for imp in blueprint.imports:
            if not imp.get("statement") or not imp.get("why"):
                return False, f"Import missing details: {imp}"

        if not blueprint.function_signature or "def " not in blueprint.function_signature:
            return False, "Function signature missing or incomplete."

        if len(blueprint.pseudocode) < 3:
            return False, (
                f"Pseudocode too shallow: {len(blueprint.pseudocode)} steps. "
                f"Need at least 3."
            )

        if not blueprint.edge_cases:
            return False, (
                "No edge cases defined. Must handle at least ImportError "
                "and general Exception."
            )

        if not blueprint.return_fields_success:
            return False, "Success return fields not defined."

        if blueprint.risk_level >= 3:
            return False, "Risk level 3 detected. This tool is too dangerous to create."

        return True, "Blueprint valid"

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 3 — CODE GENERATION FROM BLUEPRINT
    # ══════════════════════════════════════════════════════════════════════════

    async def _phase3_generate_code(
        self,
        blueprint: ToolBlueprint,
        existing_code: str | None = None,
    ) -> str | None:
        """Generate Python code by translating the blueprint.

        Args:
            blueprint: Validated ToolBlueprint.
            existing_code: Existing code to improve (if upgrading).

        Returns:
            Python source code string, or None on failure.
        """
        self.ui.show_phase_status(3, "RUNNING", "Generating code from blueprint...")

        # Build import section for prompt
        import_lines = []
        for imp in blueprint.imports:
            import_lines.append(f"  {imp['statement']}  # {imp['why']}")

        # Build pseudocode section
        pseudo_lines = []
        for i, step in enumerate(blueprint.pseudocode, 1):
            pseudo_lines.append(f"  Step {i}: {step}")

        # Build edge case section
        edge_lines = []
        for ec in blueprint.edge_cases:
            edge_lines.append(f"  - {ec['case']} → {ec['handler']}")

        # Build windows considerations
        win_lines = []
        for wc in blueprint.windows_considerations:
            win_lines.append(f"  - {wc}")

        nl = "\n"
        prompt = (
            f"BLUEPRINT:\n"
            f"Tool name: {blueprint.tool_name}\n"
            f"Task: {blueprint.task_description}\n"
            f"Approach: {blueprint.approach}\n\n"
            f"Imports (put ALL inside function body, never at top level):\n"
            f"{nl.join(import_lines)}\n\n"
            f"Function signature:\n"
            f"  {blueprint.function_signature}\n"
            f'  Docstring: "{blueprint.task_description}"\n\n'
            f"Pseudocode to implement:\n"
            f"{nl.join(pseudo_lines)}\n\n"
            f"Edge cases to handle:\n"
            f"{nl.join(edge_lines) if edge_lines else '  - Handle ImportError and general Exception'}\n\n"
            f"Return structure:\n"
            f"  Success: dict with fields: {', '.join(blueprint.return_fields_success)}\n"
            f"  Failure: dict with fields: {', '.join(blueprint.return_fields_failure)}\n\n"
            f"Windows 11 considerations:\n"
            f"{nl.join(win_lines) if win_lines else '  - Use pathlib for file paths'}\n"
        )

        if existing_code:
            prompt += (
                f"\nIMPROVE this existing code (keep what works, fix what doesn't):\n"
                f"```python\n{existing_code}\n```\n"
            )

        response = await self.llm.generate(
            prompt,
            task_type=TaskType.TOOL_CODE_GEN,
            system_prompt=_CODE_GEN_SYSTEM_PROMPT,
        )

        code = self._extract_code(response)
        if code:
            line_count = len(code.strip().splitlines())
            self.ui.show_phase_status(3, "PASS", f"{line_count} lines generated")
        else:
            logger.warning("Phase 3: failed to extract code from LLM response")

        return code

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 4 — SELF REVIEW
    # ══════════════════════════════════════════════════════════════════════════

    async def _phase4_self_review(
        self,
        code: str,
        blueprint: ToolBlueprint,
    ) -> str | None:
        """LLM reviews its own code against a checklist. Max 2 review cycles.

        Args:
            code: Python source code to review.
            blueprint: The ToolBlueprint the code should implement.

        Returns:
            Approved (possibly fixed) code, or None if still rejected after retries.
        """
        max_review_cycles = 2

        for cycle in range(max_review_cycles):
            self.ui.show_phase_status(
                4, "RUNNING",
                f"Self-review cycle {cycle + 1}/{max_review_cycles}...",
            )

            prompt = (
                f"BLUEPRINT REFERENCE:\n"
                f"  Tool: {blueprint.tool_name}\n"
                f"  Task: {blueprint.task_description}\n"
                f"  Approach: {blueprint.approach}\n"
                f"  Expected return fields (success): "
                f"{', '.join(blueprint.return_fields_success)}\n"
                f"  Expected return fields (failure): "
                f"{', '.join(blueprint.return_fields_failure)}\n\n"
                f"CODE TO REVIEW:\n"
                f"```python\n{code}\n```\n\n"
                f"Review against the checklist. If any CRITICAL issues exist, "
                f"output REJECTED then the fixed code in a ```python block. "
                f"If no CRITICAL issues, output APPROVED."
            )

            response = await self.llm.generate(
                prompt,
                task_type=TaskType.TOOL_SELF_REVIEW,
                system_prompt=_SELF_REVIEW_SYSTEM_PROMPT,
            )

            response_upper = response.upper()

            if "APPROVED" in response_upper and "REJECTED" not in response_upper:
                self.ui.show_phase_status(4, "PASS", "Self-review approved")
                return code

            if "REJECTED" in response_upper:
                # Try to extract fixed code
                fixed = self._extract_code(response)
                if fixed:
                    logger.info(
                        "Self-review cycle %d: REJECTED → applying fixes",
                        cycle + 1,
                    )
                    code = fixed
                    # If last cycle, return the fixed code anyway
                    if cycle == max_review_cycles - 1:
                        self.ui.show_phase_status(
                            4, "PASS",
                            "Applied fixes from final review cycle",
                        )
                        return code
                else:
                    # No fixed code provided, return current code
                    logger.warning(
                        "Self-review REJECTED but no fixed code provided"
                    )
                    self.ui.show_phase_status(
                        4, "PASS", "Review done (no fix code extracted)",
                    )
                    return code
            else:
                # Ambiguous response — accept current code
                self.ui.show_phase_status(4, "PASS", "Review complete")
                return code

        return code

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 5 — SANDBOX + QUALITY CHECK
    # ══════════════════════════════════════════════════════════════════════════

    async def _phase5_sandbox(
        self,
        code: str,
        function_name: str,
        param_schema: dict[str, str],
        blueprint: ToolBlueprint,
    ) -> tuple[bool, dict]:
        """Run sandbox execution with quality checking. Max 3 retries.

        Args:
            code: Python source code.
            function_name: Function to call.
            param_schema: Parameter types for dummy generation.
            blueprint: ToolBlueprint for quality checking.

        Returns:
            (success, sandbox_result) tuple.
        """
        max_sandbox_retries = 3

        for attempt in range(max_sandbox_retries):
            self.ui.show_phase_status(
                5, "RUNNING",
                f"Sandbox test {attempt + 1}/{max_sandbox_retries}...",
            )

            test_params = self.sandbox.generate_dummy_params(param_schema)
            sandbox_result = await self.sandbox.execute(
                code, function_name, test_params,
            )

            if not sandbox_result["passed"]:
                error = sandbox_result.get("error", "Unknown sandbox failure")
                logger.warning(
                    "Sandbox attempt %d failed: %s", attempt + 1, error,
                )
                if attempt < max_sandbox_retries - 1:
                    # Try to fix code using error + blueprint
                    fixed = await self._fix_code_from_error(
                        code, error, blueprint,
                    )
                    if fixed is None:
                        return False, sandbox_result
                    code = fixed
                    # Re-extract metadata in case function name changed
                    meta = self._extract_metadata(code)
                    if meta:
                        function_name = meta["name"]
                        param_schema = meta["parameters"]
                continue

            # Sandbox passed — now check result quality
            result_data = sandbox_result.get("result", {})
            quality_ok, quality_reason = self.sandbox.check_result_quality(
                result_data, blueprint,
            )

            if quality_ok:
                self.ui.show_phase_status(
                    5, "PASS", "Sandbox passed + quality check OK",
                )
                return True, sandbox_result

            logger.warning("Quality check failed: %s", quality_reason)
            if attempt < max_sandbox_retries - 1:
                fixed = await self._fix_code_from_error(
                    code, f"Quality issue: {quality_reason}", blueprint,
                )
                if fixed is None:
                    return False, sandbox_result
                code = fixed
                meta = self._extract_metadata(code)
                if meta:
                    function_name = meta["name"]
                    param_schema = meta["parameters"]

        return False, sandbox_result

    async def _fix_code_from_error(
        self,
        code: str,
        error: str,
        blueprint: ToolBlueprint,
    ) -> str | None:
        """Ask LLM to fix code based on a specific error.

        Args:
            code: Current Python code.
            error: Error message from sandbox or quality check.
            blueprint: ToolBlueprint for context.

        Returns:
            Fixed code or None on failure.
        """
        prompt = (
            f"This Python tool failed with this error:\n"
            f"ERROR: {error}\n\n"
            f"ORIGINAL TASK: {blueprint.task_description}\n"
            f"APPROACH: {blueprint.approach}\n\n"
            f"Code that failed:\n```python\n{code}\n```\n\n"
            f"Fix the specific error. Keep all imports INSIDE the function body.\n"
            f"EVERY return dict MUST have exactly these 3 keys: success (bool), result (str or None), error (str or None).\n"
            f"Output ONLY the fixed Python code in a ```python block."
        )

        response = await self.llm.generate(
            prompt,
            task_type=TaskType.TOOL_BUG_FIX,
            system_prompt=_CODE_GEN_SYSTEM_PROMPT,
        )

        return self._extract_code(response)

    # ══════════════════════════════════════════════════════════════════════════
    #  BACKWARD-COMPATIBLE API (used by decision_engine.py)
    # ══════════════════════════════════════════════════════════════════════════

    async def reason_and_plan(
        self, task_description: str,
    ) -> dict[str, Any] | None:
        """Analyze user request and produce a structured plan.

        Kept for backward compatibility with DecisionEngine.
        Internally runs Phase 1 reasoning.

        Args:
            task_description: Raw user input.

        Returns:
            Plan dict or None.
        """
        blueprint = await self._phase1_reasoning(task_description)
        if blueprint is None:
            return None

        # Convert blueprint to plan dict for compatibility
        return {
            "intent": blueprint.task_description,
            "tool_name": blueprint.tool_name,
            "description": blueprint.task_description,
            "parameters": [
                {
                    "name": p["name"],
                    "type": p["type"],
                    "desc": p.get("description", ""),
                    "required": True,
                }
                for p in blueprint.parameters
            ],
            "steps": blueprint.pseudocode,
            "edge_cases": [ec["case"] for ec in blueprint.edge_cases],
            "return_desc": ", ".join(blueprint.return_fields_success),
            "blueprint": blueprint,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_code(response: str) -> str | None:
        """Extract Python code from LLM response.

        Args:
            response: Raw LLM response string.

        Returns:
            Extracted Python code or None.
        """
        # Try ```python ... ``` first
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                pass

        # Try bare ``` ... ```
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                pass

        # Try raw response if it looks like code
        stripped = response.strip()
        if stripped.startswith(("import ", "from ", "def ")):
            try:
                ast.parse(stripped)
                return stripped
            except SyntaxError:
                pass

        return None

    @staticmethod
    def _extract_metadata(code: str) -> dict[str, Any] | None:
        """Extract function name, parameters, and docstring via AST.

        Args:
            code: Valid Python source code.

        Returns:
            Dict with name, parameters, description — or None.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                params: dict[str, str] = {}
                for arg in node.args.args:
                    type_str = "Any"
                    if arg.annotation:
                        try:
                            type_str = ast.unparse(arg.annotation)
                        except (AttributeError, ValueError):
                            if isinstance(arg.annotation, ast.Name):
                                type_str = arg.annotation.id
                            elif isinstance(arg.annotation, ast.Constant):
                                type_str = str(arg.annotation.value)
                    params[arg.arg] = type_str

                description = ""
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    description = node.body[0].value.value

                return {
                    "name": node.name,
                    "parameters": params,
                    "description": description,
                }

        return None

    @staticmethod
    def _extract_tags(description: str) -> list[str]:
        """Extract keyword tags from a task description.

        Args:
            description: Task description text.

        Returns:
            List of lowercase keyword tags (max 10).
        """
        stop_words = {
            "a", "an", "the", "to", "for", "of", "in", "on", "and", "or",
            "is", "it", "with", "from", "by", "at", "this", "that", "my",
            "me", "can", "you", "how", "what", "do", "does", "will",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", description.lower())
        return list(dict.fromkeys(w for w in words if w not in stop_words))[:10]
