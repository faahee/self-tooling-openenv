"""Decision Engine — 6-step reasoning pipeline for the self-evolving brain."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.llm_core import TaskType

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Structured specification for a tool to be synthesized."""

    name: str                                  # snake_case tool name
    description: str                           # what it does in one sentence
    inputs: dict[str, str] = field(default_factory=dict)   # param → type hint
    outputs: list[str] = field(default_factory=list)       # field names in result
    safe_libraries: list[str] = field(default_factory=list) # allowed imports
    example_call: str = ""                     # example usage string
    estimated_risk: int = 0                    # 0–3 risk level

# Existing JARVIS agents and what they handle
_EXISTING_AGENTS: dict[str, dict[str, Any]] = {
    "os_agent": {
        "description": "Launch/kill apps, volume control, screenshots, GUI automation, media playback",
        "keywords": {"open", "launch", "close", "kill", "volume", "screenshot", "play", "pause",
                     "stop", "mute", "unmute", "brightness", "app", "application", "program",
                     "spotify", "chrome", "browser", "notepad", "explorer", "settings",
                     "music", "video", "youtube", "media"},
    },
    "file_agent": {
        "description": "Read, write, search, organize, analyze files on disk",
        "keywords": {"file", "folder", "directory", "read", "write", "create", "delete",
                     "move", "copy", "rename", "search", "find", "organize", "document",
                     "resume", "pdf", "docx", "txt", "csv"},
    },
    "terminal_agent": {
        "description": "Run shell/PowerShell commands, scripts, git operations",
        "keywords": {"terminal", "command", "shell", "powershell", "cmd", "run", "execute",
                     "script", "git", "pip", "npm", "compile", "build"},
    },
    "web_agent": {
        "description": "Web search, browse websites, fetch web content",
        "keywords": {"search", "browse", "web", "website", "google", "duckduckgo", "url",
                     "link", "fetch", "scrape", "news", "weather", "online"},
    },
    "email_agent": {
        "description": "Send, read, compose emails via SMTP/IMAP",
        "keywords": {"email", "mail", "send", "inbox", "compose", "reply", "forward",
                     "smtp", "imap", "attachment"},
    },
    "image_analyzer": {
        "description": "Analyze, describe, OCR images",
        "keywords": {"image", "photo", "picture", "screenshot", "analyze", "describe",
                     "ocr", "recognize", "identify"},
    },
    "data_agent": {
        "description": "Process CSV/Excel/JSON data, charts, statistics",
        "keywords": {"data", "csv", "excel", "json", "chart", "graph", "plot",
                     "statistics", "table", "spreadsheet", "analyze"},
    },
    "scheduler_agent": {
        "description": "Schedule tasks, reminders, alarms, recurring jobs",
        "keywords": {"schedule", "remind", "reminder", "alarm", "timer", "recurring",
                     "cron", "every", "daily", "hourly", "later", "tomorrow"},
    },
    "api_agent": {
        "description": "Call external APIs — weather, crypto, GitHub, custom",
        "keywords": {"api", "weather", "crypto", "bitcoin", "github", "stock",
                     "currency", "exchange", "rate"},
    },
    "workflow_agent": {
        "description": "Multi-step goal decomposition and execution",
        "keywords": {"workflow", "steps", "chain", "sequence", "pipeline",
                     "automate", "batch"},
    },
}


class DecisionEngine:
    """6-step reasoning pipeline for the self-evolving brain.

    Every task goes through:
        Step 1: Understand user intent (handles typos, broken English, vagueness)
        Step 2: Reason about sub-steps and risks
        Step 3: Create a structured to-do list
        Step 4: Map each to-do to existing agent or decide to create a new tool
        Step 5: Synthesize new tools if needed
        Step 6: Build final execution plan and execute
    """

    # Valid agent names the LLM may reference
    _VALID_AGENTS: set[str] = set(_EXISTING_AGENTS.keys())

    # Patterns that ALWAYS route to os_agent — never synthesize a tool for these.
    # Checked before reasoning to prevent the LLM from trying to CREATE a new tool
    # for simple app/browser launches that os_agent already handles natively.
    _ALWAYS_OS_AGENT: list[tuple] = [
        (r'\b(?:open|launch|start|run)\s+(?:brave|chrome|firefox|edge|opera|vivaldi|browser)\b',
         "browser launch"),
        # "open N tabs/windows in chrome/browser" — number between open and app name
        (r'\bopen\s+\d+\s+(?:tabs?|windows?)\s+(?:in|on|with|of)\s+\w+',
         "open tabs"),
        (r'\b(?:open|launch|start)\s+(?:notepad|calculator|paint|explorer|settings|'
         r'task\s*manager|word|excel|powerpoint|outlook|teams|discord|telegram|'
         r'whatsapp|spotify|vlc|obs|steam|vscode|code)\b',
         "app launch"),
        (r'\b(?:close|kill|quit|exit)\s+(?:brave|chrome|firefox|edge|browser|'
         r'notepad|calculator|spotify|vlc|discord|telegram)\b',
         "app close"),
        (r'\btake\s+(?:a\s+)?screenshot\b', "screenshot"),
        (r'\b(?:increase|decrease|set|raise|lower|mute|unmute)\s+(?:the\s+)?volume\b',
         "volume control"),
        (r'\b(?:play|pause|stop|resume|next|previous)\s+(?:music|song|track|video)\b',
         "media control"),
    ]

    # Full reasoning prompt — forces strict JSON output
    _REASONING_SYSTEM_PROMPT: str = (
        "You are the reasoning engine of JARVIS, a personal AI OS agent.\n"
        "The user's input may have typos, broken english, slang, or be vague.\n"
        "You MUST understand the TRUE intent and never reject input.\n\n"
        "Reply with a SINGLE JSON object — no markdown, no explanation, no extra text.\n"
        "Schema:\n"
        "{\n"
        '  "understood_intent": "one clear sentence of what the user wants",\n'
        '  "assumption": "what you assumed if anything was unclear, or null",\n'
        '  "reasoning": ["thought 1", "thought 2", "..."],\n'
        '  "steps": [\n'
        '    {"order": 1, "action": "specific step description", '
        '"decision": "USE", "agent": "agent_name", "tool_name": null, '
        '"reason": "why this agent"},\n'
        '    {"order": 2, "action": "specific step description", '
        '"decision": "CREATE", "agent": null, "tool_name": "new_tool_name", '
        '"reason": "why a new tool is needed"}\n'
        "  ],\n"
        '  "fallback": null\n'
        "}\n\n"
        "Valid agents (USE only these): os_agent, file_agent, terminal_agent, "
        "web_agent, email_agent, image_analyzer, data_agent, scheduler_agent, "
        "api_agent, workflow_agent\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON — no text before or after\n"
        "- steps must have 1-8 items\n"
        "- decision is USE or CREATE, nothing else\n"
        "- USE requires agent (from the list above), tool_name must be null\n"
        "- CREATE requires tool_name, agent must be null\n"
        "- If no agent or tool can help, set steps to [] and fallback to \"direct_llm\"\n"
        "- NEVER reject broken english — always extract the real intent\n"
        "- NEVER ask clarifying questions — make assumptions and state them\n"
        "- CREATE a new Python tool when the task needs to READ SYSTEM DATA "
        "(e.g., uptime, CPU/GPU temperature, clipboard contents, running ports, "
        "disk usage, process list, network speed, RAM usage, wifi signal strength, "
        "battery status, screen resolution). These tasks need "
        "psutil/GPUtil/pyperclip/subprocess — existing agents CANNOT do this.\n"
        "- ALWAYS CREATE for: clipboard, temperature, uptime, ports, processes, "
        "speed test, system info, hardware info, network connections, RAM, CPU usage, "
        "wifi signal, wifi strength, battery level, disk space, screen info\n"
        "- USE os_agent ONLY for: launching/closing apps, volume control, "
        "screenshots, media playback (play/pause/stop music/video)\n"
        "- USE web_agent ONLY for: internet searches, browsing websites\n"
        "- USE file_agent ONLY for: reading/writing/organizing local files on disk. "
        "file_agent CANNOT access clipboard, system info, or hardware data\n"
        "- USE terminal_agent ONLY for: running shell commands the user explicitly asked for\n"
        "- When in doubt between USE and CREATE, prefer CREATE\n"
    )

    def __init__(
        self,
        config: dict,
        tool_registry,
        tool_synthesizer,
        tool_loader,
        llm_core,
        ui_layer,
        interaction_logger,
        embedder,
    ) -> None:
        self.config = config
        self.registry = tool_registry
        self.synthesizer = tool_synthesizer
        self.loader = tool_loader
        self.llm = llm_core
        self.ui = ui_layer
        self.interaction_logger = interaction_logger
        self.embedder = embedder
        self.similarity_threshold: float = config["brain"]["similarity_threshold"]
        self.reliability_threshold: float = config["brain"]["min_reliability_threshold"]
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = 3
        self.recommended_agent: str | None = None  # Set by reasoning when existing agent should handle

        # Load all existing tools on startup
        self.loader.load_all(self.registry)

    # ── Fast-path: try existing tools only ────────────────────────────

    async def try_existing_tool(self, user_input: str, threshold: float | None = None) -> str | None:
        """Check if an existing brain tool can handle this request.

        Lightweight method for CHAT fallback — no reasoning, no synthesis.
        Only searches the tool registry and runs a matching tool if found.

        Args:
            user_input: Natural language query.
            threshold: Similarity threshold override (uses config default if None).

        Returns a response string if a tool matched and ran, or None.
        """
        thresh = threshold if threshold is not None else self.similarity_threshold
        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(
            None, self.registry.search, user_input, thresh
        )
        if matches:
            best = matches[0]
            if (
                best["success_rate"] >= self.reliability_threshold
                or best["total_runs"] < 5
                or best["success_rate"] >= 0.5  # prefer imperfect tool over failing to synthesize
            ):
                logger.info(
                    "Brain fast-path: using tool '%s' (sim=%.2f)",
                    best["name"], best["similarity"],
                )
                return await self._use_tool(best, user_input)
        return None

    # ── Main entry point ─────────────────────────────────────────────

    async def handle(
        self, user_input: str, context: dict | None = None,
        force_create: bool = False,
    ) -> str | None:
        """6-step reasoning pipeline.

        Args:
            user_input: Natural language task description.
            context: Context dict from ContextBuilder.
            force_create: If True, skip reasoning and go straight to tool synthesis.
                         Used when the user explicitly says "create a tool".

        Returns a response string if handled, or None to fall back to base LLM.
        """
        # Hard override: certain tasks must ALWAYS go to os_agent.
        # Check before ANY reasoning/synthesis to prevent the LLM from
        # trying to CREATE a new tool for a simple app/browser launch.
        import re as _re
        for pattern, reason in self._ALWAYS_OS_AGENT:
            if _re.search(pattern, user_input, _re.I):
                self.recommended_agent = "os_agent"
                logger.info(
                    "ALWAYS_OS_AGENT override (%s): routing '%s' → os_agent",
                    reason, user_input[:60],
                )
                return None  # orchestrator picks up recommended_agent and routes

        # Force-create mode: user explicitly asked to "create a tool" —
        # skip reasoning (the 3B LLM keeps saying USE instead of CREATE)
        # and go straight to synthesis.
        if force_create:
            self.ui.console.print(
                "  [dim magenta]>> Direct tool synthesis (explicit create request)...[/dim magenta]"
            )
            result = await self.synthesizer.synthesize(user_input)
            if result is None:
                self._consecutive_failures += 1
                return None
            self._consecutive_failures = 0
            new_tool = self.registry.get_by_name(result["name"])
            if new_tool:
                self.loader.load(new_tool["name"], new_tool["code_path"])
                self._log_tool_event("tool_created", result["name"], result)
                return await self._use_tool(new_tool, user_input)
            return f"Tool '{result['name']}' created but could not be loaded."

        # Step 0: Quick check — is this even actionable?
        if not self._is_actionable(user_input):
            return None

        # Step 0.5: Check existing brain tools first (fast path)
        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(
            None, self.registry.search, user_input, self.similarity_threshold
        )

        if matches:
            best = matches[0]
            logger.info(
                "Brain found tool '%s' (sim=%.2f, rate=%.2f, runs=%d)",
                best["name"], best["similarity"],
                best["success_rate"], best["total_runs"],
            )

            if (
                best["success_rate"] >= self.reliability_threshold
                or best["total_runs"] < 5
            ):
                return await self._use_tool(best, user_input)

            # Low reliability — will go through full reasoning to improve
            self.ui.console.print(
                f"  [dim yellow]>> Tool '{best['name']}' has low reliability "
                f"({best['success_rate']:.0%}), re-reasoning...[/dim yellow]"
            )

        # Check consecutive failure limit
        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.warning(
                "Brain: %d consecutive failures — falling back", self._consecutive_failures
            )
            self._consecutive_failures = 0
            return None

        # ── Step 1-4: Full reasoning pipeline via LLM ──────────────
        reasoning = await self._run_reasoning(user_input)

        if reasoning:
            # Display the reasoning to the user
            self._display_reasoning(reasoning)

            # If fallback is direct_llm → return None to let base LLM handle
            if reasoning.get("fallback") == "direct_llm" and not reasoning.get("steps"):
                self.ui.console.print(
                    "  [dim yellow]>> Brain: fallback to direct LLM.[/dim yellow]"
                )
                return None

            # ── Step 4 analysis: Does anything need a new tool? ────────
            needs_creation = [
                s for s in reasoning.get("steps", [])
                if s.get("decision") == "CREATE"
            ]

            # If everything maps to existing agents → set recommended agent and return None
            # so orchestrator can re-route to the correct agent
            if not needs_creation and not matches:
                use_agents = [
                    s.get("agent", "") for s in reasoning.get("steps", [])
                    if s.get("decision") == "USE" and s.get("agent")
                    and s.get("agent") in _EXISTING_AGENTS  # only real agents
                ]
                if use_agents:
                    self.recommended_agent = use_agents[0]
                    self.ui.console.print(
                        f"  [dim green]>> Recommending: {self.recommended_agent} "
                        f"→ handing back to orchestrator.[/dim green]"
                    )
                    return None

                # No recognized agents and no tool creation planned —
                # fall through to direct synthesis below
                self.ui.console.print(
                    "  [dim yellow]>> No matching agent found, will create tool...[/dim yellow]"
                )

            # ── Step 5: Synthesize new tools ───────────────────────────
            if needs_creation:
                created_tool_name = None  # track the first successfully created tool
                for step in needs_creation:
                    tool_name = step.get("tool_name", "")
                    task_desc = step.get("action", user_input)

                    self.ui.console.print(
                        f"  [dim magenta]>> Creating new tool: {tool_name}...[/dim magenta]"
                    )

                    plan = self._reasoning_to_plan(reasoning, tool_name)
                    # Use original user_input as description (for similarity matching)
                    # but include task_desc in the plan for code gen context
                    result = await self.synthesizer.synthesize(user_input, plan=plan)

                    if result is None:
                        self._consecutive_failures += 1
                        self._log_tool_event(
                            "synthesis_failed", tool_name or user_input[:100],
                            {"attempts": self.synthesizer.max_retries},
                        )
                        continue

                    self._consecutive_failures = 0
                    if created_tool_name is None:
                        created_tool_name = result["name"]
                    new_tool = self.registry.get_by_name(result["name"])
                    if new_tool:
                        self.loader.load(new_tool["name"], new_tool["code_path"])
                        self._log_tool_event("tool_created", result["name"], result)

                # ── Step 6: Execute the first created tool ─────────────────
                if created_tool_name:
                    new_tool = self.registry.get_by_name(created_tool_name)
                    if new_tool:
                        return await self._use_tool(new_tool, user_input)
            elif not matches:
                # Reasoning didn't produce CREATE decisions and no real agents
                # matched — synthesize directly from the reasoning context
                plan = self._reasoning_to_plan(reasoning, "")
                result = await self.synthesizer.synthesize(user_input, plan=plan)
                if result:
                    self._consecutive_failures = 0
                    new_tool = self.registry.get_by_name(result["name"])
                    if new_tool:
                        self.loader.load(new_tool["name"], new_tool["code_path"])
                        self._log_tool_event("tool_created", result["name"], result)
                        return await self._use_tool(new_tool, user_input)
                else:
                    self._consecutive_failures += 1

            # If improving existing low-reliability tool
            if matches:
                best = matches[0]
                plan = self._reasoning_to_plan(reasoning, best["name"])
                improved = await self.synthesizer.synthesize(
                    best["description"], existing_tool_name=best["name"], plan=plan
                )
                if improved:
                    self._consecutive_failures = 0
                    new_tool = self.registry.get_by_name(improved["name"])
                    if new_tool:
                        self.loader.reload(new_tool["name"], new_tool["code_path"])
                        return await self._use_tool(new_tool, user_input)

        else:
            # Reasoning failed — fall back to direct synthesis (no plan)
            logger.warning("Reasoning pipeline failed, falling back to direct synthesis")
            self.ui.console.print(
                f"  [dim yellow]>> Reasoning unavailable, synthesizing directly...[/dim yellow]"
            )

            if matches:
                # Improve low-reliability tool without plan
                best = matches[0]
                improved = await self.synthesizer.synthesize(
                    best["description"], existing_tool_name=best["name"]
                )
                if improved:
                    self._consecutive_failures = 0
                    new_tool = self.registry.get_by_name(improved["name"])
                    if new_tool:
                        self.loader.reload(new_tool["name"], new_tool["code_path"])
                        return await self._use_tool(new_tool, user_input)
            else:
                # Create new tool without plan
                result = await self.synthesizer.synthesize(user_input)
                if result is None:
                    self._consecutive_failures += 1
                    self._log_tool_event(
                        "synthesis_failed", user_input[:100],
                        {"attempts": self.synthesizer.max_retries},
                    )
                    return None

                self._consecutive_failures = 0
                new_tool = self.registry.get_by_name(result["name"])
                if new_tool:
                    self.loader.load(new_tool["name"], new_tool["code_path"])
                    self._log_tool_event("tool_created", result["name"], result)
                    return await self._use_tool(new_tool, user_input)

        return None

    # ── Step 1-4: LLM Reasoning ──────────────────────────────────────

    async def _run_reasoning(self, user_input: str) -> dict[str, Any] | None:
        """Run the reasoning pipeline via LLM and return parsed JSON.

        Tries once; on JSON parse failure retries once. On second failure
        returns a fallback dict so the caller never crashes.
        """
        self.ui.console.print(
            "  [dim cyan]>> Brain reasoning (JSON pipeline)...[/dim cyan]"
        )

        prompt = (
            f'User says: "{user_input}"\n\n'
            f"Respond with the JSON object only."
        )

        for attempt in range(2):  # max 1 retry
            try:
                response = await self.llm.generate(
                    prompt,
                    task_type=TaskType.DECISION_ENGINE,
                    system_prompt=self._REASONING_SYSTEM_PROMPT
                )
            except Exception as exc:
                logger.error("Reasoning LLM call failed: %s", exc)
                return None

            parsed = self._parse_reasoning(response)
            if parsed is not None:
                return parsed

            if attempt == 0:
                logger.warning("JSON parse failed (attempt 1), retrying once")
                prompt = (
                    f'User says: "{user_input}"\n\n'
                    f"Your previous reply was not valid JSON. "
                    f"Reply with ONLY a JSON object, nothing else."
                )

        # Both attempts failed — return a minimal fallback
        logger.warning("JSON reasoning failed after 2 attempts, using fallback")
        return {
            "understood_intent": user_input,
            "assumption": None,
            "reasoning": [],
            "steps": [],
            "fallback": "direct_llm",
        }

    def _parse_reasoning(self, response: str) -> dict[str, Any] | None:
        """Parse and validate the JSON reasoning output from LLM.

        Returns a validated dict or None if the JSON is invalid / malformed.
        """
        try:
            # Try to extract JSON from the response (LLM may wrap in ```json)
            text = response.strip()
            if text.startswith("```"):
                # Strip markdown code fences
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```\s*$", "", text)

            # Find the outermost { ... }
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                logger.warning("No JSON object found in reasoning response")
                return None

            data = json.loads(text[start : end + 1])

            # ── Validate required fields ──────────────────────────
            if not isinstance(data, dict):
                return None

            # Normalise understood_intent
            intent = data.get("understood_intent", "")
            if not isinstance(intent, str) or not intent.strip():
                return None
            data["understood_intent"] = intent.strip()

            # Normalise assumption
            assumption = data.get("assumption")
            if isinstance(assumption, str) and assumption.strip().upper() == "NONE":
                assumption = None
            data["assumption"] = assumption

            # Normalise reasoning
            reasoning = data.get("reasoning", [])
            if not isinstance(reasoning, list):
                reasoning = []
            data["reasoning"] = [str(r) for r in reasoning]

            # ── Validate and fix steps ────────────────────────────
            raw_steps = data.get("steps", [])
            if not isinstance(raw_steps, list):
                raw_steps = []

            validated_steps: list[dict[str, Any]] = []
            for i, step in enumerate(raw_steps):
                if not isinstance(step, dict):
                    continue
                decision = str(step.get("decision", "")).upper()
                agent = step.get("agent")
                tool_name = step.get("tool_name")

                # Fix: USE with invalid agent → reclassify as CREATE
                if decision == "USE":
                    if not agent or agent not in self._VALID_AGENTS:
                        decision = "CREATE"
                        tool_name = agent or f"auto_tool_{i+1}"
                        agent = None

                # Fix: CREATE must have a tool_name
                if decision == "CREATE" and not tool_name:
                    tool_name = f"auto_tool_{i+1}"
                    agent = None

                # If decision is neither USE nor CREATE, guess
                if decision not in ("USE", "CREATE"):
                    decision = "USE"
                    agent = self._guess_agent(step.get("action", ""))

                validated_steps.append({
                    "order": step.get("order", i + 1),
                    "action": str(step.get("action", "")),
                    "decision": decision,
                    "agent": agent if decision == "USE" else None,
                    "tool_name": tool_name if decision == "CREATE" else None,
                    "reason": str(step.get("reason", "")),
                })

            data["steps"] = validated_steps

            # Normalise fallback
            fallback = data.get("fallback")
            if fallback not in (None, "direct_llm"):
                fallback = None
            # Empty steps with no fallback → set fallback
            if not validated_steps and fallback is None:
                fallback = "direct_llm"
            data["fallback"] = fallback

            return data

        except json.JSONDecodeError as exc:
            logger.warning("JSON decode error in reasoning: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Reasoning parse error: %s", exc)
            return None

    def _guess_agent(self, description: str) -> str:
        """Guess which agent handles a task based on keyword overlap."""
        lower = description.lower()
        words = set(re.findall(r"\b[a-z]+\b", lower))
        best_agent = "os_agent"
        best_score = 0
        for agent_name, info in _EXISTING_AGENTS.items():
            score = len(words & info["keywords"])
            if score > best_score:
                best_score = score
                best_agent = agent_name
        return best_agent

    # ── UI Display ───────────────────────────────────────────────────

    def _display_reasoning(self, reasoning: dict[str, Any]) -> None:
        """Display the JSON reasoning as a rich panel."""
        from rich.panel import Panel

        lines: list[str] = []

        # Intent
        if reasoning.get("understood_intent"):
            lines.append(f"[bold white]UNDERSTOOD:[/bold white] {reasoning['understood_intent']}")
        if reasoning.get("assumption"):
            lines.append(f"[dim white]ASSUMPTION:[/dim white] {reasoning['assumption']}")

        # Reasoning
        if reasoning.get("reasoning"):
            lines.append("")
            lines.append("[bold yellow]REASONING:[/bold yellow]")
            for r in reasoning["reasoning"]:
                lines.append(f"  [dim]•[/dim] {r}")

        # Steps (merged to-do + tool decisions)
        if reasoning.get("steps"):
            lines.append("")
            lines.append("[bold magenta]STEPS:[/bold magenta]")
            for s in reasoning["steps"]:
                order = s.get("order", "?")
                action = s.get("action", "")
                decision = s.get("decision", "?")
                if decision == "CREATE":
                    tag = f"[bold red]CREATE → {s.get('tool_name', '?')}[/bold red]"
                else:
                    tag = f"[bold green]USE → {s.get('agent', '?')}[/bold green]"
                reason = s.get("reason", "")
                lines.append(f"  [cyan]{order}.[/cyan] {action} {tag} [dim]({reason})[/dim]")

        # Fallback
        if reasoning.get("fallback"):
            lines.append("")
            lines.append(f"[bold yellow]FALLBACK:[/bold yellow] {reasoning['fallback']}")

        self.ui.console.print(
            Panel(
                "\n".join(lines),
                title="[bold cyan]🧠 Brain Reasoning Pipeline[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # ── Convert reasoning to synthesis plan ──────────────────────────

    def _reasoning_to_plan(
        self, reasoning: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        """Convert the JSON reasoning into a synthesis-compatible plan."""
        steps = [s.get("action", "") for s in reasoning.get("steps", [])]
        return {
            "intent": reasoning.get("understood_intent", ""),
            "tool_name": tool_name,
            "description": reasoning.get("understood_intent", ""),
            "parameters": [],  # LLM will figure these out during synthesis
            "steps": steps,
            "edge_cases": [
                r for r in reasoning.get("reasoning", [])
                if any(w in r.lower() for w in ("fail", "error", "wrong", "risk", "edge", "miss"))
            ],
            "return_desc": "",
        }

    # ── Tool execution ───────────────────────────────────────────────

    async def _use_tool(self, tool: dict, user_input: str) -> str:
        """Load, extract params, execute, verify, and update stats."""
        tool_name = tool["name"]

        func = self.loader.get(tool_name)
        if func is None:
            func = self.loader.load(tool_name, tool["code_path"])
            if func is None:
                self.registry.update_stats(tool_name, success=False)
                return f"Failed to load tool '{tool_name}'."

        params = await self._extract_params(user_input, tool)

        self.ui.console.print(
            f"  [dim cyan]>> Executing tool: {tool_name}[/dim cyan]"
        )
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: func(**params)
            )
        except Exception as exc:
            logger.error("Tool execution error for %s: %s", tool_name, exc)
            self.registry.update_stats(tool_name, success=False)
            return f"Tool '{tool_name}' failed during execution: {exc}"

        if not isinstance(result, dict) or "success" not in result:
            self.registry.update_stats(tool_name, success=False)
            return f"Tool '{tool_name}' returned an invalid result format."

        success = result.get("success", False)
        self.registry.update_stats(tool_name, success=success)

        if success:
            if tool_name == "music_controller":
                self._log_tool_event(
                    "tool_used", tool_name,
                    {"result": str(result.get("result", ""))[:200], "verified": True},
                )
                self.ui.console.print(
                    "  [dim green]>> Verification: skipped for direct media control tool[/dim green]"
                )
                return self._format_result(tool_name, result)
            verification = await self._verify_execution(user_input, tool_name, result)
            self._log_tool_event(
                "tool_used", tool_name,
                {"result": str(result.get("result", ""))[:200], "verified": verification["passed"]},
            )
            if not verification["passed"]:
                self.ui.console.print(
                    f"  [dim yellow]>> Verification failed: {verification['reason']}[/dim yellow]"
                )
                self.registry.update_stats(tool_name, success=False)
                return (
                    f"{self._format_result(tool_name, result)}\n\n"
                    f"⚠ Verification: {verification['reason']}"
                )
            self.ui.console.print(
                "  [dim green]>> Verification: execution successful ✓[/dim green]"
            )
            return self._format_result(tool_name, result)

        error = result.get("error", "Unknown error")
        self.ui.console.print(f"  [dim red]>> Execution failed: {error}[/dim red]")
        return f"Tool '{tool_name}' completed with error: {error}"

    # ── Post-execution verification ──────────────────────────────────

    async def _verify_execution(
        self, user_input: str, tool_name: str, result: dict
    ) -> dict[str, Any]:
        """LLM-based check that tool output actually fulfills the request."""
        result_str = str(result.get("result", ""))[:500]
        error_str = result.get("error")

        if error_str:
            return {"passed": False, "reason": f"Tool returned error: {error_str}"}
        if not result_str or result_str in ("None", "", "null", "{}"):
            return {"passed": False, "reason": "Tool returned empty or null result"}

        prompt = (
            f"You are verifying a tool's output.\n\n"
            f'User request: "{user_input}"\n'
            f"Tool: {tool_name}\n"
            f"Output: {result_str}\n\n"
            f"Does this output correctly fulfill the request?\n"
            f"Reply EXACTLY:\n"
            f"PASSED: YES or NO\n"
            f"REASON: <one sentence>\n"
        )

        try:
            response = await self.llm.generate(
                prompt,
                task_type=TaskType.INTENT_CLASSIFY
            )
            passed = "PASSED: YES" in response.upper() or "PASSED:YES" in response.upper()
            reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else (
                "Verified successfully" if passed else "May not match request"
            )
            return {"passed": passed, "reason": reason}
        except Exception as exc:
            logger.warning("Verification failed: %s — assuming passed", exc)
            return {"passed": True, "reason": "Verification skipped (LLM error)"}

    # ── Parameter extraction ─────────────────────────────────────────

    async def _extract_params(
        self, user_input: str, tool: dict
    ) -> dict[str, Any]:
        """Use LLM to extract function parameters from natural language."""
        try:
            param_schema = (
                json.loads(tool["parameters"])
                if isinstance(tool["parameters"], str)
                else tool["parameters"]
            )
        except (json.JSONDecodeError, TypeError):
            param_schema = {}

        if not param_schema:
            return {}

        prompt = (
            f"Extract parameter values from the user's request.\n\n"
            f"Function: {tool['name']}\n"
            f"Description: {tool['description']}\n"
            f"Parameters: {json.dumps(param_schema)}\n\n"
            f'User request: "{user_input}"\n\n'
            f"Return ONLY a JSON object with the parameter values.\n"
            f'Example: {{"param1": "value1", "param2": 42}}\n'
            f"No explanation, no markdown, just JSON."
        )

        response = await self.llm.generate(
            prompt,
            task_type=TaskType.INTENT_CLASSIFY
        )

        try:
            json_match = re.search(r"\{[^{}]*\}", response)
            if json_match:
                parsed = json.loads(json_match.group())
                return {k: v for k, v in parsed.items() if k in param_schema}
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.warning("Could not extract params for %s, using defaults", tool["name"])
        from brain.sandbox_executor import SandboxExecutor
        return SandboxExecutor.generate_dummy_params(param_schema)

    # ── Actionability check ──────────────────────────────────────────

    @staticmethod
    def _is_actionable(user_input: str) -> bool:
        """Determine if a request is actionable vs pure conversation."""
        lower = user_input.lower().strip()

        if len(lower.split()) < 3:
            return False

        words = set(re.findall(r"\b[a-z]+\b", lower))

        # Broad action keywords
        action_words = {
            "convert", "calculate", "compute", "generate", "create", "extract",
            "transform", "parse", "format", "sort", "filter", "merge", "split",
            "count", "encode", "decode", "hash", "encrypt", "decrypt", "validate",
            "check", "verify", "compare", "translate", "download", "fetch",
            "scrape", "summarize", "average", "sum", "multiply", "divide",
            "reverse", "replace", "clean", "normalize", "round", "truncate",
            "capitalize", "lowercase", "uppercase", "strip", "pad", "join",
            "compress", "decompress", "serialize", "deserialize", "measure",
            "benchmark", "analyze", "aggregate", "flatten", "deduplicate",
            "shuffle", "sample", "randomize", "lookup", "map", "reduce",
            "view", "show", "display", "open", "render", "preview", "list",
            "find", "search", "get", "read", "write", "save", "delete", "remove",
            "send", "post", "upload", "resize", "crop", "rotate",
            "play", "stop", "pause", "mute", "launch", "close", "kill",
            "schedule", "remind", "email", "browse", "screenshot",
            "make", "build", "setup", "install", "update", "fix", "tool",
            "copied", "clipboard", "monitor", "temperature", "uptime",
        }
        if words & action_words:
            return True

        computation_patterns = [
            r"how (?:many|much|long|far|big|old)",
            r"what is \d",
            r"what (?:is|are) (?:my|the|current)",
            r"(?:which|what) (?:app|process|program|port)",
            r"convert \d",
            r"\d+\s*(?:to|into|in)\s*\w+",
            r"(?:celsius|fahrenheit|miles|km|kg|pounds|liters|gallons)",
            r"(?:base64|hex|binary|ascii|utf|unicode)",
        ]
        for pattern in computation_patterns:
            if re.search(pattern, lower):
                return True

        return False

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_result(tool_name: str, result: dict) -> str:
        """Format a tool's result for display."""
        value = result.get("result")
        if isinstance(value, dict):
            return f"[{tool_name}] Result:\n{json.dumps(value, indent=2, default=str)}"
        if isinstance(value, (list, tuple)):
            return f"[{tool_name}] Result:\n{json.dumps(value, indent=2, default=str)}"
        return f"[{tool_name}] {value}"

    def _log_tool_event(
        self, event_type: str, tool_name: str, details: dict
    ) -> None:
        """Log a brain tool event."""
        try:
            if hasattr(self.interaction_logger, "log_tool_event"):
                self.interaction_logger.log_tool_event(event_type, tool_name, details)
        except Exception as exc:
            logger.debug("Failed to log tool event: %s", exc)

    # ══════════════════════════════════════════════════════════════════
    #  ReAct integration — plan_tool_creation
    # ══════════════════════════════════════════════════════════════════

    _PLAN_TOOL_SYSTEM_PROMPT: str = (
        "You are an expert software architect inside JARVIS (a local AI OS agent).\n"
        "Given a tool description, produce a STRICT JSON specification.\n\n"
        "Reply with ONLY a JSON object — no markdown, no explanation:\n"
        "{\n"
        '  "name": "snake_case_function_name",\n'
        '  "description": "one sentence description",\n'
        '  "inputs": {"param": "type_hint", ...},\n'
        '  "outputs": ["field1", "field2"],\n'
        '  "safe_libraries": ["json", "requests"],\n'
        '  "example_call": "tool_name(param=value)",\n'
        '  "estimated_risk": 0\n'
        "}\n\n"
        "Risk levels:\n"
        "  0 — Pure computation (math, string ops, json parsing)\n"
        "  1 — Network / file reads (requests.get, pathlib read)\n"
        "  2 — System access (pyautogui, win32api, keyboard)\n"
        "  3 — Dangerous (subprocess, shutil.rmtree, eval, exec)\n\n"
        "Rules:\n"
        "- name MUST be valid Python identifier, snake_case\n"
        "- inputs must have concrete type hints (str, int, float, list, dict)\n"
        "- outputs are field names expected inside the \"result\" key of the return dict\n"
        "- safe_libraries: ONLY from [json, os.path, pathlib, re, datetime, math, "
        "collections, hashlib, typing, requests, time, csv, base64, uuid]\n"
        "- estimated_risk MUST be 0, 1, 2, or 3\n"
        "- Output ONLY valid JSON — no text before or after\n"
    )

    async def plan_tool_creation(self, thought: Any) -> ToolSpec | None:
        """Generate a detailed tool specification from a ReAct Thought.

        Uses the LLM to produce a structured ToolSpec that guides the
        synthesizer on exactly what function to write.

        Args:
            thought: Thought dataclass from ReActEngine with intent,
                     tool_description, tool_inputs, tool_outputs.

        Returns:
            ToolSpec on success, None on failure.
        """
        description = thought.tool_description or thought.intent or ""
        if not description:
            logger.warning("plan_tool_creation: no description in thought")
            return None

        prompt = (
            f"User request: \"{description}\"\n\n"
        )
        if thought.tool_name:
            prompt += f"Suggested tool name: {thought.tool_name}\n"
        if thought.tool_inputs:
            prompt += f"Known inputs: {json.dumps(thought.tool_inputs)}\n"
        if thought.tool_outputs:
            prompt += f"Expected outputs: {thought.tool_outputs}\n"
        if thought.reasoning:
            prompt += f"Reasoning context: {thought.reasoning}\n"

        prompt += "\nGenerate the JSON tool specification."

        for attempt in range(2):
            try:
                response = await self.llm.generate(
                    prompt,
                    task_type=TaskType.TOOL_BLUEPRINT,
                    system_prompt=self._PLAN_TOOL_SYSTEM_PROMPT
                )
            except Exception as exc:
                logger.error("plan_tool_creation LLM error: %s", exc)
                return None

            spec = self._parse_tool_spec(response, description)
            if spec is not None:
                self.ui.console.print(
                    f"  [dim cyan]>> Tool spec: {spec.name} "
                    f"(risk={spec.estimated_risk}, "
                    f"inputs={list(spec.inputs.keys())}, "
                    f"outputs={spec.outputs})[/dim cyan]"
                )
                return spec

            if attempt == 0:
                logger.warning("plan_tool_creation: JSON parse failed, retrying")
                prompt = (
                    f"User request: \"{description}\"\n\n"
                    f"Your previous reply was not valid JSON. "
                    f"Reply with ONLY a JSON object, nothing else."
                )

        # Both attempts failed — build a minimal spec from what we know
        logger.warning("plan_tool_creation: falling back to minimal spec")
        fallback_name = thought.tool_name or self._make_tool_name(description)
        return ToolSpec(
            name=fallback_name,
            description=description,
            inputs=thought.tool_inputs or {},
            outputs=thought.tool_outputs or ["result"],
            safe_libraries=["json", "re", "datetime", "requests"],
            example_call=f"{fallback_name}()",
            estimated_risk=1,
        )

    def _parse_tool_spec(self, response: str, fallback_desc: str) -> ToolSpec | None:
        """Parse a ToolSpec from an LLM JSON response.

        Args:
            response: Raw LLM response string.
            fallback_desc: Description to use if the LLM omits it.

        Returns:
            ToolSpec on success, None on parse failure.
        """
        try:
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```\s*$", "", text)

            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None

            data = json.loads(text[start : end + 1])
            if not isinstance(data, dict):
                return None

            name = data.get("name", "")
            if not name or not re.match(r"^[a-z][a-z0-9_]*$", name):
                name = self._make_tool_name(fallback_desc)

            description = data.get("description", fallback_desc) or fallback_desc

            inputs = data.get("inputs", {})
            if not isinstance(inputs, dict):
                inputs = {}

            outputs = data.get("outputs", ["result"])
            if not isinstance(outputs, list):
                outputs = ["result"]

            safe_libs = data.get("safe_libraries", ["json", "requests"])
            if not isinstance(safe_libs, list):
                safe_libs = ["json", "requests"]

            example = data.get("example_call", f"{name}()")
            if not isinstance(example, str):
                example = f"{name}()"

            risk = data.get("estimated_risk", 1)
            if not isinstance(risk, int) or risk not in (0, 1, 2, 3):
                risk = 1

            return ToolSpec(
                name=name,
                description=description,
                inputs=inputs,
                outputs=outputs,
                safe_libraries=safe_libs,
                example_call=example,
                estimated_risk=risk,
            )
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("ToolSpec parse error: %s", exc)
            return None

    @staticmethod
    def _make_tool_name(description: str) -> str:
        """Generate a snake_case tool name from a description.

        Args:
            description: Natural language description.

        Returns:
            Valid snake_case Python identifier.
        """
        words = re.findall(r"[a-zA-Z]+", description.lower())
        stop = {"a", "an", "the", "to", "for", "of", "in", "on", "and", "or",
                "is", "it", "with", "from", "by", "at", "me", "my", "i",
                "that", "this", "what", "how", "can", "you", "do", "show"}
        filtered = [w for w in words if w not in stop][:5]
        name = "_".join(filtered) if filtered else "auto_tool"
        return f"{name}_tool"
