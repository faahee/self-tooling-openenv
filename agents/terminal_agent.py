"""Terminal Agent — Claude Code-style autonomous terminal execution.

Handles any terminal/coding task: running commands, creating/editing files,
installing packages, git operations, and multi-step project scaffolding.

Uses direct pattern matching for instant execution of simple commands,
and an LLM-driven planning loop for complex multi-step tasks.
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Output strings that indicate a command needs elevated privileges
_ELEVATION_ERRORS = (
    "access is denied",
    "requires elevation",
    "run as administrator",
    "administrator privileges",
    "elevated",
    "access denied",
    "privilege",
    "0x80070005",           # E_ACCESSDENIED HRESULT
    "is not recognized",    # winget/choco not on PATH for non-admin
    "cannot be loaded because running scripts is disabled",
    "you do not have permission",
    "insufficient privileges",
)

# Dangerous command patterns — blocked before execution
_BLOCKED_PATTERNS = [
    r"format\s+[a-z]:",
    r"del\s+/[sfq].*[a-z]:\\",
    r"rm\s+-rf\s+/\s*$",
    r"rmdir\s+/s\s+/q\s+[a-z]:\\$",
    r"reg\s+delete",
    r"bcdedit",
    r"diskpart",
    r"shutdown\s+/[rpf]",
    r"sfc\s+/",
    r"dism\s+/",
]

# First-word prefixes that always indicate a terminal command
COMMAND_PREFIXES: set[str] = {
    # Python
    "python", "python3", "py", "pip", "pip3", "pipx", "poetry", "uv",
    # Node
    "node", "npm", "npx", "yarn", "pnpm", "bun", "deno",
    # Git
    "git",
    # Containers
    "docker", "docker-compose", "podman",
    # Rust / Go / C / Java / .NET
    "cargo", "rustc", "go", "gcc", "g++", "make", "cmake",
    "java", "javac", "mvn", "gradle", "dotnet",
    # Package managers
    "conda", "mamba", "choco", "scoop", "winget",
    # Shell utilities
    "ls", "dir", "cd", "pwd", "mkdir", "rmdir", "cat", "type",
    "echo", "touch", "cp", "mv", "rm", "head", "tail",
    "grep", "glob", "find", "findstr", "where", "which",
    "curl", "wget", "ssh", "scp",
    # JARVIS extended patterns
    "explain", "fix", "debug", "review",
    # System info
    "ping", "ipconfig", "netstat", "nslookup", "tracert",
    # Build / test
    "pytest", "unittest", "jest", "mocha", "tox", "nox",
}


class TerminalAgent:
    """Claude Code-style autonomous terminal agent.

    Capable of:
      - Direct execution of shell commands (pip, git, python, etc.)
      - Creating and editing files
      - Multi-step project scaffolding via LLM planning loop
      - Reading file contents and directory listings
      - Maintaining persistent working directory across commands
    """

    # ── Claude Code-style structured tool protocol ──────────────────────────
    # The LLM outputs exactly ONE action per turn using these keywords.
    # Each action maps to a native Python method — no shell for file I/O.
    AGENT_SYSTEM_PROMPT = (
        "You are JARVIS, a Claude Code-style autonomous coding agent on Windows 11 with PowerShell.\n"
        "Faheem is your boss. Complete tasks autonomously, step by step.\n\n"
        "=== TOOL PROTOCOL ===\n"
        "Output exactly ONE action per reply. Choose from:\n\n"
        "1. Read a file (ALWAYS do this before editing):\n"
        "   READ: path/to/file.py\n\n"
        "2. Read specific lines only:\n"
        "   READ: path/to/file.py LINES: 1-50\n\n"
        "3. Edit an existing file (precise replacement — never rewrite whole file):\n"
        "   EDIT: path/to/file.py\n"
        "   OLD: exact text to replace (copy from file)\n"
        "   ---\n"
        "   NEW: replacement text\n\n"
        "4. Write a new file:\n"
        "   WRITE: path/to/file.py\n"
        "   ```\n"
        "   file content here\n"
        "   ```\n\n"
        "5. Run a shell command:\n"
        "   ```powershell\n"
        "   command here\n"
        "   ```\n\n"
        "6. Search file content:\n"
        "   GREP: pattern IN path/to/file.py\n\n"
        "7. Find files by pattern:\n"
        "   GLOB: *.py\n\n"
        "8. Signal completion:\n"
        "   DONE: one-line summary of what was accomplished\n\n"
        "=== RULES ===\n"
        "- ONE action per reply — never combine multiple actions\n"
        "- Always READ a file before EDIT — never guess content\n"
        "- Use EDIT (not WRITE) for existing files — preserve unchanged lines\n"
        "- If a command fails, READ the error carefully and try a different approach\n"
        "- After 3 failures on the same approach, try something completely different\n"
        "- When task is complete: DONE: <summary>"
    )

    MAX_ITERATIONS = 20          # increased for multi-file tasks
    COMMAND_TIMEOUT = 180        # seconds per shell command
    MAX_OUTPUT_CHARS = 4000      # truncation limit for command output
    AGENT_MODEL = "qwen2.5-coder:7b"  # routed via LLMCore TaskType.SHELL_CMD_GEN

    def __init__(self, config: dict, llm_core, ui_layer, checkpoint_manager=None) -> None:
        """Initialize the terminal agent.

        Args:
            config: Full JARVIS configuration dictionary.
            llm_core: LLMCore instance for LLM planning loop.
            ui_layer: UILayer instance for Rich terminal display.
            checkpoint_manager: Optional CheckpointManager for undo support.
        """
        self.config = config
        self.llm = llm_core
        self.ui = ui_layer
        self.checkpoint = checkpoint_manager
        self.cwd = Path.cwd()
        self._last_exit_code: int = 0
        logger.info("TerminalAgent ready. Working directory: %s", self.cwd)

    # ── MAIN ENTRY POINT ─────────────────────────────────────────────────────

    async def execute(self, user_input: str, context: dict) -> str:
        """Handle any terminal/coding request.

        Tries direct pattern matching first for instant execution.
        Falls back to an LLM-driven multi-step planning loop for
        complex tasks (project scaffolding, file editing, etc.).

        Args:
            user_input: The user's natural language or raw command.
            context: Context dict from ContextBuilder.

        Returns:
            Result string describing what was done.
        """
        # Fast path — direct pattern matching
        direct = self._try_direct(user_input)
        if direct is not None:
            return direct

        # Slow path — LLM agent loop for complex tasks
        return await self._agent_loop(user_input, context)

    # ── COMMAND PREFIX DETECTION ──────────────────────────────────────────────

    @staticmethod
    def looks_like_command(text: str) -> bool:
        """Check if text looks like a raw terminal command.

        Args:
            text: User input text.

        Returns:
            True if the first word is a known command prefix.
        """
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        return first_word in COMMAND_PREFIXES

    # ── DIRECT EXECUTION (no LLM needed) ─────────────────────────────────────

    def _try_direct(self, text: str) -> str | None:
        """Pattern-match common commands for instant execution.

        Returns result string, or None if no pattern matched.
        """
        nl = text.lower().strip()
        original = text.strip()

        # ── Navigation ────────────────────────────────────────────
        m = re.match(r"cd\s+(.+)", nl)
        if m:
            return self._change_dir(m.group(1).strip())

        if nl in ("ls", "dir", "ll"):
            return self._run_cmd(
                "Get-ChildItem | Format-Table Mode, LastWriteTime, Length, Name"
            )

        if nl in ("pwd", "cwd", "where am i"):
            return f"Working directory: {self.cwd}"

        # ── Directory creation ────────────────────────────────────
        m = re.match(
            r"(?:mkdir|make directory|create folder|create directory|make folder)\s+(.+)",
            nl,
        )
        if m:
            name = m.group(1).strip().strip("\"'")
            return self._run_cmd(f'New-Item -ItemType Directory -Name "{name}" -Force')

        # ── Package managers ──────────────────────────────────────
        m = re.match(r"(pip|pip3|pipx|poetry|uv)\s+(.+)", nl)
        if m:
            return self._run_cmd(f"{m.group(1)} {m.group(2)}")

        m = re.match(r"(npm|npx|yarn|pnpm|bun)\s+(.+)", nl)
        if m:
            return self._run_cmd(f"{m.group(1)} {m.group(2)}")

        m = re.match(r"(conda|mamba)\s+(.+)", nl)
        if m:
            return self._run_cmd(f"{m.group(1)} {m.group(2)}")

        # "install X" (no prefix) → pip install X
        m = re.match(r"install\s+(?:package\s+)?(.+)", nl)
        if m:
            pkg = m.group(1).strip()
            return self._run_cmd(f"pip install {pkg}")

        # ── Git ───────────────────────────────────────────────────
        m = re.match(r"git\s+(.+)", nl)
        if m:
            return self._run_cmd(f"git {m.group(1)}")

        # ── Language runtimes ─────────────────────────────────────
        m = re.match(r"(python|python3|py)\s+(.+)", nl)
        if m:
            return self._run_cmd(f"python {m.group(2)}")

        m = re.match(r"(node|deno|bun)\s+(.+)", nl)
        if m:
            return self._run_cmd(f"{m.group(1)} {m.group(2)}")

        # Build / test runners
        m = re.match(r"(pytest|jest|mocha|tox|nox|make|cmake|cargo|go|dotnet|mvn|gradle)\s*(.*)", nl)
        if m:
            cmd = m.group(1)
            args = m.group(2).strip()
            return self._run_cmd(f"{cmd} {args}" if args else cmd)

        # ── Docker ────────────────────────────────────────────────
        m = re.match(r"(docker|docker-compose|podman)\s+(.+)", nl)
        if m:
            return self._run_cmd(f"{m.group(1)} {m.group(2)}")

        # ── File readers ──────────────────────────────────────────
        m = re.match(r"(?:cat|type|read|show|view)\s+(.+)", nl)
        if m:
            return self._read_file(m.group(1).strip())

        # ── Run / execute (generic) ───────────────────────────────
        m = re.match(r"(?:run|execute)\s+(.+)", nl)
        if m:
            cmd = m.group(1).strip()
            # If it looks like "run my python script" → try python
            if "python" in cmd or cmd.endswith(".py"):
                script = re.sub(r"^(?:my\s+)?python\s+(?:script\s+)?", "", cmd).strip()
                return self._run_cmd(f"python {script}" if script else "python")
            return self._run_cmd(cmd)

        # ── Network utils ─────────────────────────────────────────
        m = re.match(r"(ping|ipconfig|netstat|nslookup|tracert|curl|wget|ssh)\s*(.*)", nl)
        if m:
            cmd = m.group(1)
            args = m.group(2).strip()
            return self._run_cmd(f"{cmd} {args}" if args else cmd)

        # ── Grep: search file content ──────────────────────────────
        m = re.match(r"(?:grep|search in|find in|search content)\s+(.+?)\s+(?:in|inside|for)\s+(.+)", nl)
        if m:
            pattern, location = m.group(1).strip(), m.group(2).strip()
            return self._run_cmd(
                f"Select-String -Path '{location}' -Pattern '{pattern}' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 30"
            )
        # grep <pattern> (searches current dir)
        m = re.match(r"grep\s+(.+)", nl)
        if m:
            return self._run_cmd(
                f"Select-String -Path '.' -Pattern '{m.group(1).strip()}' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 30"
            )

        # ── Glob: find files by pattern ────────────────────────────
        m = re.match(r"(?:glob|find files?|list files?)\s+(.+)", nl)
        if m:
            pat = m.group(1).strip().strip("\"'")
            return self._run_cmd(f"Get-ChildItem -Recurse -Filter '{pat}' -ErrorAction SilentlyContinue | Select-Object FullName")

        # ── Explain file: read and describe ───────────────────────
        m = re.match(r"(?:explain|describe|summarize|what does)\s+(.+?)(?:\s+do|mean|contain)?\s*$", nl)
        if m:
            target = m.group(1).strip().strip("\"'")
            path = (self.cwd / target).resolve()
            if path.is_file():
                return self._read_file(str(path))
            # Fall through to LLM agent loop for conceptual questions

        # ── Fix / review file: read it so the LLM can work ────────
        m = re.match(r"(?:fix|debug|review|check|lint|improve)\s+(.+)", nl)
        if m:
            target = m.group(1).strip().strip("\"'")
            path = (self.cwd / target).resolve()
            if path.is_file():
                # Read the file and return its content so the LLM agent loop
                # starts with the file already in context
                content = self._read_file(str(path))
                return None  # Let agent loop handle it with the printed context

        # ── Catch-all: if the first word is a known command ───────
        first_word = nl.split()[0] if nl else ""
        if first_word in COMMAND_PREFIXES:
            return self._run_cmd(original)

        return None

    # ── SHELL EXECUTION ──────────────────────────────────────────────────────

    def _is_blocked(self, command: str) -> bool:
        """Check if a command matches any blocked pattern."""
        cl = command.lower()
        return any(re.search(pat, cl) for pat in _BLOCKED_PATTERNS)

    def _run_cmd(self, command: str, silent: bool = False) -> str:
        """Run a shell command via PowerShell and return formatted output.

        Args:
            command: The command string to execute.
            silent: If True, don't print to the Rich console.

        Returns:
            Combined stdout+stderr output, or error message.
            If the output contains elevation errors, the caller should
            call _run_cmd_elevated() after getting user approval.
        """
        if self._is_blocked(command):
            msg = f"[!] Blocked dangerous command: {command}"
            logger.warning(msg)
            return msg

        if not silent:
            self.ui.console.print(f"\n  [bold cyan]> Running:[/bold cyan] [white]{command}[/white]")
            self.ui.console.print(f"  [dim]  cwd: {self.cwd}[/dim]")

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=self.COMMAND_TIMEOUT,
                cwd=str(self.cwd),
            )
            self._last_exit_code = result.returncode

            output = ""
            if result.stdout.strip():
                output += result.stdout.strip()
            if result.stderr.strip():
                if output:
                    output += "\n"
                output += result.stderr.strip()

            if not output:
                output = "(no output)"

            # Truncate long output
            if len(output) > self.MAX_OUTPUT_CHARS:
                output = (
                    output[: self.MAX_OUTPUT_CHARS]
                    + f"\n... (truncated, {len(output)} total chars)"
                )

            if not silent:
                if result.returncode == 0:
                    self.ui.console.print(f"  [bold green][OK][/bold green] Exit code 0")
                else:
                    self.ui.console.print(
                        f"  [bold red][FAIL][/bold red] Exit code {result.returncode}"
                    )
                # Print output with indentation
                for line in output.split("\n")[:50]:
                    self.ui.console.print(f"    {line}")
                if output.count("\n") > 50:
                    self.ui.console.print(f"    ... ({output.count(chr(10))} lines total)")

            return output

        except subprocess.TimeoutExpired:
            msg = f"[!] Command timed out after {self.COMMAND_TIMEOUT}s: {command}"
            if not silent:
                self.ui.console.print(f"  [bold red]{msg}[/bold red]")
            return msg
        except Exception as exc:
            msg = f"[!] Error running command: {exc}"
            if not silent:
                self.ui.console.print(f"  [bold red]{msg}[/bold red]")
            return msg

    # ── FILE OPERATIONS ──────────────────────────────────────────────────────

    def _change_dir(self, target: str) -> str:
        """Change the persistent working directory.

        Args:
            target: Relative or absolute path.

        Returns:
            Confirmation or error string.
        """
        target = target.strip("\"'")

        if target == "~":
            new_path = Path.home()
        elif target == "-":
            new_path = self.cwd.parent
        else:
            new_path = (self.cwd / target).resolve()

        if new_path.is_dir():
            self.cwd = new_path
            self.ui.console.print(f"  [bold green][OK] cd -> {self.cwd}[/bold green]")
            return f"Changed directory to: {self.cwd}"

        return f"Directory not found: {target}"

    def _read_file(self, filepath: str) -> str:
        """Read a file's content.

        Args:
            filepath: Relative or absolute path.

        Returns:
            File content string, or error message.
        """
        filepath = filepath.strip("\"'")
        target = (self.cwd / filepath).resolve()

        if not target.is_file():
            return f"File not found: {filepath}"

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > self.MAX_OUTPUT_CHARS:
                content = content[: self.MAX_OUTPUT_CHARS] + "\n... (truncated)"
            self.ui.console.print(f"\n  [bold cyan]FILE: {filepath}[/bold cyan]")
            for i, line in enumerate(content.split("\n")[:60], 1):
                self.ui.console.print(f"  [dim]{i:>4}[/dim] | {line}")
            if content.count("\n") > 60:
                self.ui.console.print(f"  [dim]  ... ({content.count(chr(10))+1} lines total)[/dim]")
            return content
        except Exception as exc:
            return f"Error reading file: {exc}"

    def _write_file(self, filepath: str, content: str) -> str:
        """Create or overwrite a file.

        Args:
            filepath: Relative or absolute path.
            content: File content string.

        Returns:
            Confirmation or error string.
        """
        filepath = filepath.strip("\"'")
        target = (self.cwd / filepath).resolve()
        # Snapshot before overwriting so the user can undo
        if self.checkpoint:
            self.checkpoint.snapshot(str(target), description=f"before write: {filepath}")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            self.ui.console.print(
                f"  [bold green][OK] Written:[/bold green] {filepath} ({lines} lines)"
            )
            return f"File written: {filepath} ({lines} lines)"
        except Exception as exc:
            return f"Error writing file: {exc}"

    def _edit_file(self, filepath: str, old_text: str, new_text: str) -> str:
        """Replace text in an existing file.

        Args:
            filepath: Relative or absolute path.
            old_text: Exact text to find.
            new_text: Replacement text.

        Returns:
            Confirmation or error string.
        """
        filepath = filepath.strip("\"'")
        target = (self.cwd / filepath).resolve()

        if not target.is_file():
            return f"File not found: {filepath}"

        # Snapshot before editing
        if self.checkpoint:
            self.checkpoint.snapshot(str(target), description=f"before edit: {filepath}")

        try:
            content = target.read_text(encoding="utf-8")
            if old_text not in content:
                return f"Text not found in {filepath}. Edit failed."
            new_content = content.replace(old_text, new_text, 1)
            target.write_text(new_content, encoding="utf-8")
            self.ui.console.print(
                f"  [bold green][OK] Edited:[/bold green] {filepath}"
            )
            return f"File edited: {filepath}"
        except Exception as exc:
            return f"Error editing file: {exc}"

    # ── LLM AGENT LOOP ──────────────────────────────────────────────────────

    async def _agent_loop(self, user_input: str, context: dict) -> str:
        """Multi-step LLM-driven execution loop — Claude Code style.

        The LLM emits one structured tool call per turn (READ / EDIT / WRITE /
        RUN / GREP / GLOB / DONE).  We dispatch it natively in Python, feed
        the result back, and repeat until DONE or the iteration cap.

        Args:
            user_input: The user's natural language request.
            context: Context dict from ContextBuilder.

        Returns:
            Summary string of what was accomplished.
        """
        self.ui.console.print("\n  [bold yellow]>> Starting agentic task loop...[/bold yellow]")

        # Track files that have been READ in this session (read-before-edit)
        self._read_files: set[str] = set()

        messages = [
            {"role": "system", "content": self.AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Working directory: {self.cwd}\n\n"
                    f"Task: {user_input}"
                ),
            },
        ]

        all_steps: list[str] = []
        consecutive_failures = 0

        from core.llm_core import TaskType as _TaskType

        for iteration in range(self.MAX_ITERATIONS):
            self.ui.console.print(
                f"\n  [dim]── Step {iteration + 1}/{self.MAX_ITERATIONS} ──[/dim]"
            )

            response = await self.llm.chat(
                messages,
                task_type=_TaskType.SHELL_CMD_GEN,
                max_tokens=1000,
                temperature=0.1,
            )

            if not response or not response.strip():
                break

            logger.debug("Agent step %d: %r", iteration + 1, response[:200])

            # ── Dispatch the tool call ────────────────────────────
            tool_name, result, step_label = self._dispatch_tool(response)

            if tool_name == "DONE":
                self.ui.console.print(
                    f"\n  [bold green][DONE][/bold green] {result}"
                )
                return result

            if tool_name == "UNKNOWN":
                # LLM didn't follow protocol — nudge it
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    self.ui.console.print(
                        "\n  [bold red][FAIL] LLM not following tool protocol — stopping.[/bold red]"
                    )
                    break
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "I couldn't parse your action. Use exactly one of: "
                        "READ: / EDIT: / WRITE: / GREP: / GLOB: / ```powershell / DONE:"
                    ),
                })
                continue

            # Append action + result to conversation
            messages.append({"role": "assistant", "content": response})
            all_steps.append(step_label)

            success = tool_name in ("READ", "GLOB", "GREP", "WRITE", "EDIT") or self._last_exit_code == 0

            # ── Elevation intercept ───────────────────────────────
            # If a RUN command failed with an access/privilege error,
            # ask the user before counting it as a failure.
            if tool_name == "RUN" and not success and self._needs_elevation(result):
                cmd_match = re.search(r"^> Running.*?:\s*(.+)$", result, re.MULTILINE)
                failed_cmd = cmd_match.group(1).strip() if cmd_match else step_label.replace("Run: ", "")
                approved = await self._ask_elevation(failed_cmd)
                if approved:
                    elevated_result = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_cmd_elevated, failed_cmd
                    )
                    result = elevated_result
                    success = "[!]" not in elevated_result
                    self._last_exit_code = 0 if success else 1
                    all_steps[-1] = f"Run elevated: {failed_cmd[:60]}"
                else:
                    result += "\n[User declined elevation.]"

            if success:
                consecutive_failures = 0
                messages.append({
                    "role": "user",
                    "content": (
                        f"Result:\n{result[:2000]}\n\n"
                        "Continue. Use the next tool action, or DONE: <summary> if finished."
                    ),
                })
            else:
                consecutive_failures += 1
                self.ui.console.print(
                    f"  [bold red][FAIL][/bold red] Exit code {self._last_exit_code}"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"Result (FAILED — exit code {self._last_exit_code}):\n{result[:2000]}\n\n"
                        "The command failed. Read the error carefully. "
                        "Try a completely different approach, or DONE: <reason> if unrecoverable."
                    ),
                })
                if consecutive_failures >= 3:
                    self.ui.console.print(
                        "\n  [bold red][FAIL] 3 consecutive command failures — stopping.[/bold red]"
                    )
                    break

            # Keep context window manageable (system + first user + last N turns)
            if len(messages) > 14:
                messages = messages[:2] + messages[-10:]

        # Loop exhausted
        if all_steps:
            summary = f"Completed {len(all_steps)} steps: " + "; ".join(
                s for s in all_steps[:6]
            )
            self.ui.console.print(f"\n  [bold green][OK][/bold green] {summary}")
            return summary

        return "I couldn't determine what actions to take for that request."

    # ── STRUCTURED TOOL DISPATCHER ────────────────────────────────────────────

    def _dispatch_tool(self, response: str) -> tuple[str, str, str]:
        """Parse and execute one structured tool call from an LLM response.

        Returns:
            (tool_name, result_string, step_label)
            tool_name is one of: READ, EDIT, WRITE, RUN, GREP, GLOB, DONE, UNKNOWN
        """
        text = response.strip()

        # ── DONE ─────────────────────────────────────────────────
        m = re.match(r"DONE[:\s]+(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return "DONE", m.group(1).strip(), "Done"

        # Bare DONE
        if text.strip().lower() in ("done", "done."):
            return "DONE", "Task complete.", "Done"

        # ── READ: path [LINES: start-end] ─────────────────────────
        m = re.match(
            r"READ:\s*(.+?)(?:\s+LINES?:\s*(\d+)\s*-\s*(\d+))?\s*$",
            text, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            filepath = m.group(1).strip()
            line_start = int(m.group(2)) if m.group(2) else None
            line_end = int(m.group(3)) if m.group(3) else None
            result = self._read_file_ranged(filepath, line_start, line_end)
            self._read_files.add(filepath)
            return "READ", result, f"Read {filepath}"

        # ── EDIT: path\nOLD: ...\n---\nNEW: ... ──────────────────
        m = re.match(
            r"EDIT:\s*(.+?)\n(?:OLD:\s*)?(.*?)\n---\n(?:NEW:\s*)?(.*)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            filepath = m.group(1).strip()
            old_text = m.group(2).strip()
            new_text = m.group(3).strip()
            if filepath not in self._read_files:
                # Auto-read first so the edit is informed
                self.ui.console.print(
                    f"  [dim yellow]  Auto-reading {filepath} before edit...[/dim yellow]"
                )
                self._read_file_ranged(filepath)
                self._read_files.add(filepath)
            result = self._edit_file(filepath, old_text, new_text)
            return "EDIT", result, f"Edit {filepath}"

        # ── WRITE: path\n```\ncontent\n``` ────────────────────────
        m = re.match(
            r"WRITE:\s*(.+?)\n```(?:\w+)?\s*\n(.*?)\n```",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            filepath = m.group(1).strip()
            content = m.group(2)
            result = self._write_file(filepath, content)
            self._read_files.add(filepath)
            return "WRITE", result, f"Write {filepath}"

        # ── GREP: pattern IN path ─────────────────────────────────
        m = re.match(r"GREP:\s*(.+?)\s+IN\s+(.+)", text, re.IGNORECASE)
        if m:
            pattern, location = m.group(1).strip(), m.group(2).strip()
            result = self._run_cmd(
                f"Select-String -Path '{location}' -Pattern '{pattern}' "
                f"-Recurse -ErrorAction SilentlyContinue | Select-Object -First 40",
                silent=True,
            )
            self.ui.console.print(
                f"  [bold cyan]GREP[/bold cyan] {pattern!r} in {location}"
            )
            return "GREP", result, f"Grep '{pattern}' in {location}"

        # ── GLOB: pattern ─────────────────────────────────────────
        m = re.match(r"GLOB:\s*(.+)", text, re.IGNORECASE)
        if m:
            pattern = m.group(1).strip().strip("\"'")
            result = self._run_cmd(
                f"Get-ChildItem -Recurse -Filter '{pattern}' "
                f"-ErrorAction SilentlyContinue | Select-Object FullName | "
                f"Select-Object -First 30",
                silent=True,
            )
            self.ui.console.print(f"  [bold cyan]GLOB[/bold cyan] {pattern}")
            return "GLOB", result, f"Glob {pattern}"

        # ── RUN: ```powershell / ```bash / inline code block ──────
        cmd = self._extract_command(text)
        if cmd:
            result = self._run_cmd(cmd)
            return "RUN", result, f"Run: {cmd[:60]}"

        return "UNKNOWN", "", "Unknown action"

    # ── ELEVATION SUPPORT ────────────────────────────────────────────────────

    @staticmethod
    def _needs_elevation(output: str) -> bool:
        """Return True if command output signals an elevation requirement."""
        lower = output.lower()
        return any(err in lower for err in _ELEVATION_ERRORS)

    def _run_cmd_elevated(self, command: str) -> str:
        """Re-run a PowerShell command with UAC elevation.

        Writes the command to a temp .ps1 script, launches it via
        ShellExecuteW with the 'runas' verb (triggers UAC dialog),
        captures stdout/stderr to a temp output file, and reads it back.

        Args:
            command: PowerShell command string to execute elevated.

        Returns:
            Combined stdout+stderr output string, or error message.
        """
        tmp = Path(tempfile.gettempdir())
        script_path = tmp / "jarvis_elevated_cmd.ps1"
        out_path = tmp / "jarvis_elevated_out.txt"

        # Clean up any previous run
        out_path.unlink(missing_ok=True)

        # Write temp script — captures all output to out_path
        script_path.write_text(
            "$ErrorActionPreference = 'Continue'\n"
            f"({command}) 2>&1 | Out-File -FilePath '{out_path}' -Encoding UTF8\n",
            encoding="utf-8",
        )

        self.ui.console.print(
            "  [bold yellow]⚡ UAC prompt opening — please click Yes...[/bold yellow]"
        )

        # Launch elevated via ShellExecuteW — triggers Windows UAC dialog
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            "powershell.exe",
            f'-ExecutionPolicy Bypass -NoProfile -NonInteractive -File "{script_path}"',
            str(self.cwd),
            0,  # SW_HIDE — no extra window
        )

        if ret <= 32:
            # ShellExecute error codes ≤ 32 indicate failure
            _SE_ERRORS = {
                2: "file not found",
                3: "path not found",
                5: "access denied (UAC cancelled?)",
                8: "out of memory",
                31: "no application associated",
                32: "DLL not found",
            }
            reason = _SE_ERRORS.get(ret, f"error code {ret}")
            return f"[!] Elevation failed: {reason}"

        # Poll for output file — elevated process writes async
        self.ui.console.print(
            "  [dim]  Waiting for elevated command to complete...[/dim]"
        )
        for attempt in range(90):   # wait up to 90 seconds
            time.sleep(1)
            if out_path.exists() and out_path.stat().st_size > 0:
                try:
                    output = out_path.read_text(encoding="utf-8", errors="replace")
                    if output.strip():
                        out_path.unlink(missing_ok=True)
                        script_path.unlink(missing_ok=True)
                        self._last_exit_code = 0
                        if not self.ui:
                            return output.strip()
                        self.ui.console.print(
                            "  [bold green][OK] Elevated command completed[/bold green]"
                        )
                        for line in output.strip().split("\n")[:40]:
                            self.ui.console.print(f"    {line}")
                        return output.strip()
                except Exception:
                    pass

        return (
            "[!] Elevated command launched but output not captured within 90s. "
            "Check if the UAC prompt was accepted and the command completed."
        )

    async def _ask_elevation(self, command: str) -> bool:
        """Prompt the user to approve running a command with admin rights.

        Args:
            command: The command that needs elevation.

        Returns:
            True if the user approved, False otherwise.
        """
        self.ui.console.print(
            f"\n  [bold yellow]⚡ Elevation Required[/bold yellow]\n"
            f"  Command: [white]{command[:120]}[/white]\n"
            f"  This command needs Administrator rights.\n"
        )
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: input("  Allow elevation? (yes/no): "),
        )
        return answer.strip().lower() in ("yes", "y", "ok", "sure", "allow", "1")

    # ── RANGED FILE READ ──────────────────────────────────────────────────────

    def _read_file_ranged(
        self, filepath: str, line_start: int | None = None, line_end: int | None = None
    ) -> str:
        """Read a file, optionally limiting to a line range.

        Args:
            filepath: Relative or absolute path.
            line_start: First line to include (1-based), or None for start.
            line_end: Last line to include (1-based), or None for end.

        Returns:
            File content string with line numbers, or error message.
        """
        filepath = filepath.strip("\"'")
        target = (self.cwd / filepath).resolve()

        if not target.is_file():
            return f"File not found: {filepath}"

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)

            start = max(0, (line_start or 1) - 1)
            end = min(total, line_end or total)
            selected = lines[start:end]

            # Truncate if still too long
            if sum(len(l) for l in selected) > self.MAX_OUTPUT_CHARS:
                selected = selected[: self.MAX_OUTPUT_CHARS // 80]
                selected.append("... (truncated — use LINES: x-y for a smaller range)")

            label = f"{filepath} (lines {start+1}-{start+len(selected)} of {total})"
            self.ui.console.print(f"\n  [bold cyan]READ:[/bold cyan] {label}")
            for i, line in enumerate(selected[:80], start + 1):
                self.ui.console.print(f"  [dim]{i:>4}[/dim] | {line}")
            if len(selected) > 80:
                self.ui.console.print(f"  [dim]  ... {len(selected)-80} more lines[/dim]")

            return "\n".join(f"{start+i+1}: {l}" for i, l in enumerate(selected))
        except Exception as exc:
            return f"Error reading file: {exc}"

    # ── STREAMING SHELL EXECUTION ─────────────────────────────────────────────

    def _run_cmd_streaming(self, command: str) -> str:
        """Run a shell command and stream output line-by-line to the console.

        Used for long-running commands (installs, builds, tests).
        Falls back to buffered _run_cmd for commands that finish quickly.

        Args:
            command: PowerShell command string.

        Returns:
            Full output string (same as _run_cmd).
        """
        if self._is_blocked(command):
            msg = f"[!] Blocked dangerous command: {command}"
            logger.warning(msg)
            return msg

        self.ui.console.print(
            f"\n  [bold cyan]> Running (streaming):[/bold cyan] [white]{command}[/white]"
        )

        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.cwd),
            )
            lines: list[str] = []
            for line in proc.stdout:  # type: ignore[union-attr]
                stripped = line.rstrip()
                lines.append(stripped)
                self.ui.console.print(f"    {stripped}")
                if len(lines) > 200:
                    self.ui.console.print("    ... (output truncated at 200 lines)")
                    proc.kill()
                    break
            proc.wait(timeout=self.COMMAND_TIMEOUT)
            self._last_exit_code = proc.returncode
            return "\n".join(lines)
        except subprocess.TimeoutExpired:
            msg = f"[!] Command timed out after {self.COMMAND_TIMEOUT}s"
            self.ui.console.print(f"  [bold red]{msg}[/bold red]")
            return msg
        except Exception as exc:
            msg = f"[!] Error: {exc}"
            self.ui.console.print(f"  [bold red]{msg}[/bold red]")
            return msg

    # ── LEGACY HELPERS (kept for _is_done / _extract_summary / _extract_command)

    def _is_done(self, response: str) -> bool:
        """Check if the LLM response signals task completion."""
        text = response.strip().lower()
        if text in ("done", "done."):
            return True
        if text.startswith("done") and "```" not in text and "`" not in text:
            return True
        return False

    def _contains_done(self, response: str) -> bool:
        """Check if the response contains 'done' alongside a command."""
        text = response.strip().lower()
        return bool(re.search(r"\bdone\b", text))

    def _extract_summary(self, response: str, steps: list[str]) -> str:
        """Extract a summary from a DONE response, or build one from steps."""
        m = re.search(r"done[:\s]*(.+)", response.strip(), re.IGNORECASE)
        if m:
            summary = m.group(1).strip().strip("\"'.")
            if len(summary) > 10:
                return summary
        if steps:
            return f"Completed {len(steps)} step(s): " + "; ".join(
                s.replace("Ran: ", "") for s in steps[:5]
            )
        return "Task completed."

    def _extract_command(self, response: str) -> str | None:
        """Extract a shell command from an LLM response.

        Handles:
          - Code blocks (```...```)
          - ACTION: RUN / COMMAND: patterns
          - Bare commands on their own line
        """
        text = response.strip()
        clean = text.replace("\\\\", "\\").replace('\\"', '"')

        # ── Code block (most common Phi-3 output) ────────────────
        m = re.search(
            r"```(?:powershell|bash|cmd|shell|ps1|python)?\s*\n(.+?)\n\s*```",
            clean,
            re.DOTALL,
        )
        if m:
            cmd = m.group(1).strip()
            # Filter out comment-only lines
            lines = [l.strip() for l in cmd.split("\n")
                     if l.strip() and not l.strip().startswith("#")]
            if lines:
                return "; ".join(lines)

        # ── ACTION: RUN / COMMAND: pattern ────────────────────────
        m = re.search(r"ACTION:\s*RUN\s*\n\s*COMMAND:\s*(.+?)(?:\n|$)", clean)
        if m:
            return m.group(1).strip()
        m = re.search(r"ACTION:\s*RUN\s+COMMAND[:\s]+(.+?)(?:\n|$)", clean)
        if m:
            return m.group(1).strip()
        m = re.search(r"^COMMAND:\s*(.+?)(?:\n|$)", clean, re.MULTILINE)
        if m:
            return m.group(1).strip()

        # ── Inline code: `command here` ──────────────────────────
        m = re.search(r"`([^`]{3,})`", clean)
        if m:
            candidate = m.group(1).strip()
            if candidate and not candidate.startswith("http"):
                return candidate

        # ── Single bare line that looks like a command ────────────
        lines = [l.strip() for l in clean.split("\n") if l.strip()]
        if len(lines) == 1 and self.looks_like_command(lines[0]):
            return lines[0]

        return None

