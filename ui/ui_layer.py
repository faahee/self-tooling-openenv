"""UI Layer — Terminal display (Rich), TTS voice output, and Windows notifications."""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import subprocess
import tempfile
import threading
import winsound

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

# Intent → colour mapping
INTENT_COLORS: dict[str, str] = {
    "TERMINAL_TASK": "bright_green",
    "OS_COMMAND": "bright_cyan",
    "FILE_OP": "bright_green",
    "IMAGE_ANALYZE": "bright_magenta",
    "MEMORY_QUERY": "bright_yellow",
    "CODE_TASK": "bright_blue",
    "WEB_TASK": "orange1",
    "SYSTEM_QUERY": "bright_white",
    "BRAIN_TASK": "bright_magenta",
    "CHAT": "white",
    "CACHE_HIT": "dim",
}


class UILayer:
    """All JARVIS-to-user output: streaming terminal, TTS, and notifications."""

    def __init__(self, config: dict) -> None:
        """Initialize the UI layer.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.console = Console()
        self.tts_enabled: bool = config.get("ui", {}).get("tts_enabled", True)
        self.stream_display: bool = config["ui"]["stream_display"]

        # TTS engine — runs in its own thread on Windows
        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_rate: int = config["voice"]["tts_rate"]
        self._tts_volume: float = config["voice"]["tts_volume"]
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._tts_thread.start()

        # Streaming state
        self._stream_buffer: str = ""

    # ── Streaming display ─────────────────────────────────────────────────────

    def stream_token(self, token: str) -> None:
        """Display a single streamed token from the LLM.

        Args:
            token: The token string to display.
        """
        self._stream_buffer += token
        self.console.print(token, end="", highlight=False)

    def end_stream(self) -> None:
        """Finish the current streaming output."""
        if self._stream_buffer:
            self.console.print()  # newline
            self._stream_buffer = ""

    # ── Full response display ─────────────────────────────────────────────────

    def display_response(self, response: str, intent: str | None = None) -> None:
        """Display a complete response.

        If the response was already streamed via stream_token(), just
        finish the stream (newline) — don't re-print the full text.
        Otherwise show it in a Rich Panel.

        Args:
            response: The full response text.
            intent: Optional intent label for colour-coding.
        """
        if self._stream_buffer:
            # Already printed via streaming — just close cleanly
            self.end_stream()
            return

        color = INTENT_COLORS.get(intent or "", "white")
        panel = Panel(
            Text(response, style=color),
            title="[bold bright_blue]JARVIS[/bold bright_blue]",
            border_style="bright_blue",
            padding=(0, 1),
        )
        self.console.print(panel)

    def display_user_input(self, text: str, source: str = "text") -> None:
        """Show the user's input.

        Args:
            text: What the user said/typed.
            source: "voice" or "text".
        """
        icon = "🎤" if source == "voice" else "⌨"
        self.console.print(f"\n[bold green]{icon} You:[/bold green] {text}")

    # ── TTS ───────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Queue text for TTS (non-blocking).

        Args:
            text: The text to speak aloud.
        """
        if self.tts_enabled and text:
            logger.debug("TTS queued: %s", text[:80])
            self._tts_queue.put(text)

    def _tts_loop(self) -> None:
        """Background thread: generate speech via SAPI5 PowerShell, play via winsound.

        Uses a separate PowerShell process for SAPI5 to avoid COM threading issues,
        and winsound for playback (doesn't conflict with sounddevice recording).
        """
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                self._speak_via_sapi(text)
            except Exception as exc:
                logger.error("TTS error: %s", exc)

    def _speak_via_sapi(self, text: str) -> None:
        """Generate speech WAV via PowerShell SAPI5, then play via winsound."""
        # Strip to plain ASCII-safe text for TTS (avoids PS escaping issues)
        import re
        safe_text = re.sub(r"[\r\n]+", " ", text)
        safe_text = safe_text[:500]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as txt_tmp:
            txt_tmp.write(safe_text)
            txt_path = txt_tmp.name

        try:
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$synth.Rate = {max(-10, min(10, (self._tts_rate - 175) // 20))}; "
                f"$synth.Volume = {int(self._tts_volume * 100)}; "
                f"$synth.SetOutputToWaveFile('{wav_path}'); "
                f"$text = [System.IO.File]::ReadAllText('{txt_path}'); "
                "$synth.Speak($text); "
                "$synth.Dispose()"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.error("TTS PowerShell error: %s", result.stderr[:300])
                return

            # Play WAV through Windows default audio (doesn't conflict with sd.rec())
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            try:
                os.unlink(txt_path)
            except OSError:
                pass

    # ── Notifications ─────────────────────────────────────────────────────────

    def notify(self, title: str, message: str) -> None:
        """Show a Windows toast notification.

        Args:
            title: Notification title.
            message: Notification body.
        """
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name="JARVIS",
                timeout=5,
            )
        except Exception as exc:
            logger.debug("Notification error: %s", exc)

    # ── Validator approval ─────────────────────────────────────────────────────

    async def prompt_import_approval(self, module_name: str) -> bool:
        """Prompt the user to approve a Level 2 (system-access) import.

        Called by CodeValidator.validate_async() when a generated tool
        wants to use a Level 2 module (e.g. ctypes, pyautogui).

        Args:
            module_name: The module requesting approval.

        Returns:
            True if the user approves, False otherwise.
        """
        self.console.print(
            f"\n[bold yellow]⚠  IMPORT APPROVAL REQUIRED[/bold yellow]"
        )
        self.console.print(
            f"  Tool wants to import [bold]{module_name}[/bold] "
            f"(Level 2 — system access)"
        )
        self.console.print("  [dim]Auto-reject in 30 seconds[/dim]")

        loop = asyncio.get_event_loop()
        response: str = await loop.run_in_executor(
            None,
            lambda: input("  Allow this import? (yes/no): ").strip().lower(),
        )
        return response in ("yes", "y")

    # ── Memory Auditor report ────────────────────────────────────────────────

    def show_audit_report(self, report: Any) -> None:
        """Display a memory audit report in the terminal.

        Args:
            report: AuditReport instance with a .summary property and .to_dict().
        """
        summary = report.summary
        self.console.print(
            f"\n[bold bright_cyan]🧹 [JARVIS] Memory audit complete:[/bold bright_cyan] "
            f"[cyan]{summary}[/cyan]"
        )

        # Show details if anything was cleaned
        data = report.to_dict()
        if data.get("tools_deactivated"):
            for name in data["tools_deactivated"]:
                self.console.print(
                    f"  [dim red]✗ Deactivated: {name}[/dim red]"
                )
        if data.get("tools_archived"):
            for name in data["tools_archived"]:
                self.console.print(
                    f"  [dim yellow]↪ Archived: {name}[/dim yellow]"
                )
        if data.get("orphan_files_deleted"):
            for name in data["orphan_files_deleted"]:
                self.console.print(
                    f"  [dim red]✗ Orphan deleted: {name}[/dim red]"
                )
        if data.get("duration_seconds"):
            self.console.print(
                f"  [dim]Completed in {data['duration_seconds']:.2f}s[/dim]"
            )

    # ── Session summary ─────────────────────────────────────────────────────

    def show_session_summary(self, summary: Any) -> None:
        """Display the session learning summary on shutdown.

        Args:
            summary: SessionSummary instance with a .format() method.
        """
        self.console.print()
        self.console.print(
            Panel(
                Text(summary.format(), style="bright_cyan"),
                title="[bold bright_blue]Session Summary[/bold bright_blue]",
                border_style="bright_blue",
                padding=(0, 1),
            )
        )
        if summary.lessons_learned:
            self.console.print(
                f"  [bold bright_green]>> {summary.lessons_learned} lesson(s) "
                f"saved for next session[/bold bright_green]"
            )
        if summary.top_lesson:
            self.console.print(
                f"  [dim cyan]Top lesson: {summary.top_lesson}[/dim cyan]"
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    def show_status(self, message: str) -> None:
        """Show a status/info line.

        Args:
            message: Status message.
        """
        self.console.print(f"[dim cyan][i] {message}[/dim cyan]")

    def show_error(self, message: str) -> None:
        """Show an error message in red.

        Args:
            message: Error message.
        """
        self.console.print(f"[bold red][X] Error: {message}[/bold red]")

    def show_banner(self) -> None:
        """Display the JARVIS startup banner."""
        banner = r"""
       _ _____  ____  __   __ ___  ____
      | |  _  ||  _ \\|  | /  /|_ ||  __|
      | | |_| || |_) )| |/  /  | || |__
  _   | |  _  ||    / |    \   | ||__ \\
 | |__| | | | || |\\ \\|  |\  \ | | __| |
  \____/|_| |_||_| \_\|__| \__\|_||____|
    """
        self.console.print(f"[bold bright_blue]{banner}[/bold bright_blue]")
        self.console.print(
            "[bold]Personal AI OS Agent[/bold] -- "
            "[dim]Fully local - RTX 3050 optimised - Windows 11[/dim]\n"
        )

    def stop(self) -> None:
        """Shut down the TTS thread."""
        self._tts_queue.put(None)

    # ── Synthesis progress display ────────────────────────────────────────────

    def show_synthesis_header(self, tool_name: str) -> None:
        """Display the tool creation header panel.

        Args:
            tool_name: Name of the tool being created.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold bright_magenta]🔨 Tool Creation: {tool_name}[/bold bright_magenta]",
                border_style="bright_magenta",
                padding=(0, 1),
            )
        )

    def show_blueprint(self, blueprint: object) -> None:
        """Display the reasoning blueprint summary.

        Args:
            blueprint: ToolBlueprint instance with approach, pseudocode,
                       edge_cases, imports, risk_level, confidence.
        """
        approach = getattr(blueprint, "approach", "")
        steps = getattr(blueprint, "pseudocode", [])
        edge_cases = getattr(blueprint, "edge_cases", [])
        imports = getattr(blueprint, "imports", [])
        risk = getattr(blueprint, "risk_level", 0)
        confidence = getattr(blueprint, "confidence", 0)

        lines = [
            f"  [bold]Approach:[/bold] {approach}",
            f"  [bold]Steps:[/bold] {len(steps)}  |  "
            f"[bold]Edge cases:[/bold] {len(edge_cases)}  |  "
            f"[bold]Imports:[/bold] {len(imports)}",
            f"  [bold]Risk:[/bold] {risk}  |  "
            f"[bold]Confidence:[/bold] {confidence:.2f}",
        ]

        for i, step in enumerate(steps, 1):
            lines.append(f"    {i}. [dim]{step}[/dim]")

        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold]Phase 1 — Blueprint[/bold]",
                border_style="bright_magenta",
                padding=(0, 1),
            )
        )

    def show_phase_status(
        self, phase: int, status: str, detail: str = "",
    ) -> None:
        """Display a single phase status line.

        Args:
            phase: Phase number (1-5).
            status: Status string (RUNNING, PASS, RETRY, REGISTERED, etc.).
            detail: Optional detail text.
        """
        phase_names = {
            1: "REASONING",
            2: "BLUEPRINT VALIDATION",
            3: "CODE GENERATION",
            4: "SELF REVIEW",
            5: "SANDBOX + QUALITY",
        }
        name = phase_names.get(phase, f"PHASE {phase}")

        if status in ("PASS", "REGISTERED"):
            icon = "✅"
            style = "bold green"
        elif status == "RUNNING":
            icon = "⏳"
            style = "bold yellow"
        elif status == "RETRY":
            icon = "🔄"
            style = "bold yellow"
        else:
            icon = "❌"
            style = "bold red"

        msg = f"  [{style}]{icon} PHASE {phase} — {name}: {status}[/{style}]"
        if detail:
            msg += f"  [dim]({detail})[/dim]"
        self.console.print(msg)

    def show_synthesis_failure(self, reason: str) -> None:
        """Display a tool synthesis failure message.

        Args:
            reason: Human-readable failure reason.
        """
        self.console.print(
            f"  [bold red]❌ SYNTHESIS FAILED: {reason}[/bold red]"
        )

    # ── ReAct trace display ───────────────────────────────────────────────────

    def show_react_trace(
        self,
        iteration: int,
        thought: Any,
        result: Any,
        observation: Any,
        elapsed_ms: int,
    ) -> None:
        """Display the full ReAct loop trace for one iteration.

        Shows the REASON → ACT → OBSERVE pipeline with rich formatting,
        including the CREATE path with synthesis, validation, sandbox,
        save, and registration steps.

        Args:
            iteration: Current loop iteration number.
            thought: Thought dataclass from ReActEngine.
            result: ActionResult dataclass.
            observation: Observation dataclass.
            elapsed_ms: Total elapsed milliseconds.
        """
        lines: list[str] = []

        # ── REASON ────────────────────────────────────────────────
        path = getattr(thought, "path", "?")
        reasoning = getattr(thought, "reasoning", "")
        lines.append(
            f"[bold white]🧠 REASON :[/bold white] {reasoning}"
        )
        lines.append(
            f"   [dim]Path   : {path}[/dim]"
        )

        if path == "AGENT":
            agent = getattr(thought, "agent", "?")
            confidence = getattr(thought, "confidence", 0)
            lines.append(
                f"   [dim]Agent  : {agent} (confidence {confidence:.2f})[/dim]"
            )

        elif path == "TOOL":
            tool_name = getattr(thought, "tool_name", "?")
            confidence = getattr(thought, "confidence", 0)
            lines.append(
                f"   [dim]Tool   : {tool_name} (sim {confidence:.2f})[/dim]"
            )

        elif path == "CREATE":
            tool_name = getattr(thought, "tool_name", "?") or "new_tool"
            lines.append(
                f"   [dim]Tool   : {tool_name}[/dim]"
            )

        lines.append("")

        # ── ACT ───────────────────────────────────────────────────
        if result.success:
            lines.append(
                f"[bold green]⚡ ACT    :[/bold green] "
                f"{'executed successfully' if path != 'CREATE' else 'tool created and executed'} "
                f"({result.duration_ms}ms)"
            )
        else:
            error_short = (result.error or "unknown error")[:80]
            lines.append(
                f"[bold red]⚡ ACT    :[/bold red] failed — {error_short}"
            )

        # Show CREATE-specific steps from metadata
        metadata = getattr(result, "metadata", {}) or {}
        if metadata.get("newly_created"):
            code_path = metadata.get("code_path", "")
            version = metadata.get("version", 1)
            if code_path:
                lines.append(
                    f"   [dim]💾 SAVED  : {code_path}[/dim]"
                )
            lines.append(
                f"   [dim]📋 REGISTER: brain.db updated (v{version})[/dim]"
            )

        lines.append("")

        # ── OBSERVE ──────────────────────────────────────────────
        quality = getattr(observation, "quality", 0)
        complete = getattr(observation, "complete", False)
        feedback = getattr(observation, "feedback", "")

        if complete:
            lines.append(
                f"[bold green]👁 OBSERVE:[/bold green] "
                f"complete=True quality={quality:.2f}"
            )
        else:
            lines.append(
                f"[bold yellow]👁 OBSERVE:[/bold yellow] "
                f"complete=False — {feedback}"
            )

        missing = getattr(observation, "missing_fields", [])
        if missing:
            lines.append(
                f"   [dim]Missing : {', '.join(missing)}[/dim]"
            )

        self.console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold cyan]ReAct Loop (iteration {iteration})[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    async def prompt_tool_approval(self, tool_spec: Any) -> bool:
        """Prompt the user to approve creation of a risk-level-2 tool.

        Shows tool name, description, risk level, and libraries.
        Auto-denies after 30 seconds if no response.

        Args:
            tool_spec: ToolSpec dataclass with name, description,
                       estimated_risk, safe_libraries.

        Returns:
            True if approved, False if denied or timeout.
        """
        name = getattr(tool_spec, "name", "unknown_tool")
        desc = getattr(tool_spec, "description", "")
        risk = getattr(tool_spec, "estimated_risk", 2)
        libs = getattr(tool_spec, "safe_libraries", [])

        self.console.print()
        self.console.print(
            f"[bold yellow]⚠  TOOL CREATION APPROVAL REQUIRED[/bold yellow]"
        )
        self.console.print(
            f"  Tool    : [bold]{name}[/bold]"
        )
        self.console.print(
            f"  Desc    : {desc}"
        )
        self.console.print(
            f"  Risk    : Level {risk} (system access)"
        )
        if libs:
            self.console.print(
                f"  Libraries: {', '.join(libs)}"
            )
        self.console.print("  [dim]Auto-reject in 30 seconds[/dim]")

        loop = asyncio.get_event_loop()
        try:
            response: str = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: input("  Allow this tool? (yes/no): ").strip().lower(),
                ),
                timeout=30.0,
            )
            return response in ("yes", "y")
        except (asyncio.TimeoutError, EOFError):
            self.console.print(
                "  [dim red]>> Auto-rejected (timeout)[/dim red]"
            )
            return False
