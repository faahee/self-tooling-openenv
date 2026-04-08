"""ReAct Engine — Reason-Act-Observe loop with self-evolving tool creation."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Thought:
    """Structured reasoning output from the REASON step."""

    intent: str
    path: str                             # "AGENT", "TOOL", "CREATE"
    agent: str | None = None              # set if path == "AGENT"
    tool_name: str | None = None          # set if path == "TOOL" or "CREATE"
    tool_description: str | None = None   # set if path == "CREATE"
    tool_inputs: dict = field(default_factory=dict)
    tool_outputs: list[str] = field(default_factory=list)
    action_params: dict = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    skip_fields: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ActionResult:
    """Result of an ACT step."""

    success: bool
    output: Any = None
    error: str | None = None
    tool_name: str | None = None
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether the action produced a usable result."""
        return self.success and self.output is not None


@dataclass
class Observation:
    """Result of the OBSERVE step."""

    complete: bool
    quality: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    feedback: str = ""


# ── Agent keyword map ─────────────────────────────────────────────────────────

_AGENT_KEYWORDS: dict[str, set[str]] = {
    "os_agent": {
        "open", "launch", "close", "kill", "volume", "screenshot", "play",
        "pause", "stop", "mute", "brightness", "app", "application",
        "spotify", "chrome", "browser", "notepad", "explorer", "settings",
        "music", "video", "youtube", "media", "edge", "tab", "window",
    },
    "file_agent": {
        "file", "folder", "directory", "read", "write", "create", "delete",
        "move", "copy", "rename", "search", "find", "organize", "document",
        "resume", "pdf", "docx", "txt", "csv",
    },
    "terminal_agent": {
        "terminal", "command", "shell", "powershell", "cmd", "run",
        "execute", "script", "git", "pip", "npm", "compile", "build",
    },
    "web_agent": {
        "search", "browse", "web", "website", "google", "duckduckgo",
        "url", "link", "fetch", "scrape", "online",
    },
    "email_agent": {
        "email", "mail", "send", "inbox", "compose", "reply", "forward",
        "smtp", "imap", "attachment",
    },
    "image_analyzer": {
        "image", "photo", "picture", "screenshot", "analyze", "describe",
        "ocr", "recognize", "identify",
    },
    "data_agent": {
        "data", "csv", "excel", "json", "chart", "graph", "plot",
        "statistics", "table", "spreadsheet",
    },
    "scheduler_agent": {
        "schedule", "remind", "reminder", "alarm", "timer", "recurring",
        "cron", "every", "daily", "hourly", "later", "tomorrow",
    },
    "api_agent": {
        "api", "weather", "crypto", "bitcoin", "github", "stock",
        "currency", "exchange", "rate", "joke", "ip",
        "temperature", "forecast", "rain", "humidity", "wind",
        "price", "news", "headline", "usd", "inr", "eur",
        "ethereum", "eth", "btc", "solana", "dogecoin",
    },
    "workflow_agent": {
        "workflow", "steps", "chain", "sequence", "pipeline",
        "automate", "batch",
    },
}


class ReActEngine:
    """Reason-Act-Observe loop with three execution paths.

    Path A — Existing agent handles the request directly.
    Path B — Existing tool from the brain registry is loaded and executed.
    Path C — No agent or tool exists; the brain creates, validates,
             sandboxes, registers, and executes a brand-new tool.

    Hard limits:
        - Max 5 iterations per request
        - Max 3 synthesis attempts per tool (delegated to synthesizer)
        - All LLM calls go through llm_core
    """

    def __init__(
        self,
        config: dict,
        llm_core: Any,
        tool_registry: Any,
        tool_synthesizer: Any,
        tool_loader: Any,
        decision_engine: Any,
        code_validator: Any,
        sandbox_executor: Any,
        ui_layer: Any,
        orchestrator: Any,
        interaction_logger: Any,
    ) -> None:
        """Initialise the ReAct engine.

        Args:
            config: Full JARVIS configuration dictionary.
            llm_core: LLMCore instance for all LLM calls.
            tool_registry: ToolRegistry instance (brain.db).
            tool_synthesizer: ToolSynthesizer instance.
            tool_loader: ToolLoader for dynamic import of tool files.
            decision_engine: DecisionEngine for tool spec planning.
            code_validator: CodeValidator for AST risk checks.
            sandbox_executor: SandboxExecutor for safe test runs.
            ui_layer: UILayer for display and approval prompts.
            orchestrator: Orchestrator reference for agent routing.
            interaction_logger: InteractionLogger for event logging.
        """
        self.config = config
        self.llm = llm_core
        self.registry = tool_registry
        self.synthesizer = tool_synthesizer
        self.loader = tool_loader
        self.decision_engine = decision_engine
        self.validator = code_validator
        self.sandbox = sandbox_executor
        self.ui = ui_layer
        self.orchestrator = orchestrator
        self.interaction_logger = interaction_logger

        brain_cfg = config.get("brain", {})
        react_cfg = config.get("react", {})
        self._max_iterations: int = react_cfg.get("max_iterations", 5)
        self._similarity_threshold: float = brain_cfg.get("similarity_threshold", 0.75)
        self._reliability_threshold: float = brain_cfg.get("min_reliability_threshold", 0.7)

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════════

    async def run(self, user_input: str, context: dict) -> str:
        """Execute the full ReAct loop for a user request.

        Args:
            user_input: Natural language request.
            context: Context dict from ContextBuilder.

        Returns:
            Final response string.
        """
        start_time = time.time()

        for iteration in range(1, self._max_iterations + 1):
            logger.info("ReAct iteration %d/%d", iteration, self._max_iterations)

            # ── REASON ────────────────────────────────────────────────
            thought = await self._reason(user_input, context)
            logger.info(
                "Thought: path=%s agent=%s tool=%s confidence=%.2f",
                thought.path, thought.agent, thought.tool_name, thought.confidence,
            )

            # ── ACT ──────────────────────────────────────────────────
            if thought.path == "AGENT":
                result = await self._act_agent(thought, user_input, context)
            elif thought.path == "TOOL":
                result = await self._act_tool(thought, user_input)
            elif thought.path == "CREATE":
                result = await self._act_create(thought, user_input)
            else:
                result = ActionResult(
                    success=False, error=f"Unknown path: {thought.path}"
                )

            # ── OBSERVE ──────────────────────────────────────────────
            observation = self._observe(thought, result)

            # ── DISPLAY TRACE ────────────────────────────────────────
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.ui.show_react_trace(
                iteration=iteration,
                thought=thought,
                result=result,
                observation=observation,
                elapsed_ms=elapsed_ms,
            )

            if observation.complete:
                self._log_event(
                    "react_complete",
                    user_input,
                    {
                        "iterations": iteration,
                        "path": thought.path,
                        "tool": thought.tool_name,
                        "quality": observation.quality,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                return str(result.output)

            # Feed observation back for next iteration
            user_input = (
                f"{user_input}\n\n[Previous attempt feedback: {observation.feedback}]"
            )

        # Exhausted iterations
        logger.warning("ReAct loop exhausted %d iterations", self._max_iterations)
        return (
            "I tried multiple approaches but could not complete this request. "
            "Could you rephrase or provide more details?"
        )

    # ══════════════════════════════════════════════════════════════════
    #  REASON — decide which path to take
    # ══════════════════════════════════════════════════════════════════

    async def _reason(self, user_input: str, context: dict) -> Thought:
        """Determine the best execution path for a request.

        Priority order:
            1. Keyword match → existing agent      → path = AGENT
            2. Registry search (sim > threshold,
               success_rate > reliability)          → path = TOOL
            3. Registry search (sim > threshold,
               success_rate < reliability)          → path = CREATE (improve)
            4. Nothing found                        → path = CREATE (new)

        Args:
            user_input: Natural language request.
            context: Context dict from ContextBuilder.

        Returns:
            Thought dataclass with the chosen path and details.
        """
        lower = user_input.lower()
        words = set(re.findall(r"\b[a-z]+\b", lower))

        # ── 1. Check existing agents by keyword overlap ───────────────
        best_agent: str | None = None
        best_score = 0
        for agent_name, keywords in _AGENT_KEYWORDS.items():
            score = len(words & keywords)
            if score > best_score:
                best_score = score
                best_agent = agent_name

        # api_agent queries are often single-keyword ("temperature", "bitcoin")
        # so we accept score >= 1 for it; all other agents require >= 2.
        min_score = 1 if best_agent == "api_agent" else 2
        if best_agent and best_score >= min_score:
            return Thought(
                intent=user_input,
                path="AGENT",
                agent=best_agent,
                confidence=min(1.0, best_score / 5.0),
                reasoning=f"Keyword match ({best_score} hits) → {best_agent}",
            )

        # ── 2 & 3. Semantic search in tool registry ──────────────────
        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(
            None, self.registry.search, user_input, self._similarity_threshold
        )

        if matches:
            best = matches[0]
            sim = best["similarity"]
            rate = best["success_rate"]
            runs = best["total_runs"]

            if rate >= self._reliability_threshold or runs < 5:
                # Path B — reliable existing tool
                return Thought(
                    intent=user_input,
                    path="TOOL",
                    tool_name=best["name"],
                    tool_description=best["description"],
                    confidence=sim,
                    reasoning=(
                        f"Registry match: '{best['name']}' "
                        f"(sim={sim:.2f}, rate={rate:.0%}, runs={runs})"
                    ),
                )
            else:
                # Path C — improve unreliable tool
                return Thought(
                    intent=user_input,
                    path="CREATE",
                    tool_name=best["name"],
                    tool_description=best["description"],
                    confidence=sim,
                    reasoning=(
                        f"Tool '{best['name']}' found but unreliable "
                        f"(rate={rate:.0%}, runs={runs}) → re-synthesize"
                    ),
                )

        # ── 4. Nothing found → brand new tool ────────────────────────
        return Thought(
            intent=user_input,
            path="CREATE",
            tool_description=user_input,
            confidence=0.0,
            reasoning="No existing agent or tool matches → creating new tool",
        )

    # ══════════════════════════════════════════════════════════════════
    #  ACT — Path A: route to existing agent
    # ══════════════════════════════════════════════════════════════════

    async def _act_agent(
        self, thought: Thought, user_input: str, context: dict
    ) -> ActionResult:
        """Route request to an existing JARVIS agent.

        Args:
            thought: The reasoning output.
            user_input: Original user input.
            context: Context dict from ContextBuilder.

        Returns:
            ActionResult from the agent.
        """
        agent_name = thought.agent or ""
        intent = self.orchestrator._agent_to_intent(agent_name)
        if not intent:
            return ActionResult(
                success=False,
                error=f"No intent mapping for agent '{agent_name}'",
            )

        start = time.time()
        try:
            history = self.orchestrator._build_history_snippet()
            response = await self.orchestrator.route_to_agent(
                intent, user_input, context, history
            )
            elapsed = int((time.time() - start) * 1000)
            return ActionResult(
                success=True,
                output=response,
                duration_ms=elapsed,
                metadata={"agent": agent_name, "intent": intent},
            )
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            logger.error("Agent %s failed: %s", agent_name, exc)
            return ActionResult(
                success=False,
                error=str(exc),
                duration_ms=elapsed,
            )

    # ══════════════════════════════════════════════════════════════════
    #  ACT — Path B: load & execute existing brain tool
    # ══════════════════════════════════════════════════════════════════

    async def _act_tool(
        self, thought: Thought, user_input: str
    ) -> ActionResult:
        """Load and execute an existing tool from the registry.

        Args:
            thought: The reasoning output (must have tool_name).
            user_input: Original user input.

        Returns:
            ActionResult from tool execution.
        """
        tool_name = thought.tool_name or ""
        tool = self.registry.get_by_name(tool_name)
        if not tool:
            return ActionResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
            )

        func = self.loader.get(tool_name)
        if func is None:
            func = self.loader.load(tool_name, tool["code_path"])
            if func is None:
                self.registry.update_stats(tool_name, success=False)
                return ActionResult(
                    success=False,
                    error=f"Failed to load tool '{tool_name}'",
                    tool_name=tool_name,
                )

        # Extract parameters via LLM
        params = await self.decision_engine._extract_params(user_input, tool)

        start = time.time()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: func(**params)
            )
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            logger.error("Tool %s execution error: %s", tool_name, exc)
            self.registry.update_stats(tool_name, success=False)
            return ActionResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                duration_ms=elapsed,
            )

        elapsed = int((time.time() - start) * 1000)

        if not isinstance(result, dict) or "success" not in result:
            self.registry.update_stats(tool_name, success=False)
            return ActionResult(
                success=False,
                error="Tool returned invalid result format",
                tool_name=tool_name,
                duration_ms=elapsed,
            )

        success = result.get("success", False)
        self.registry.update_stats(tool_name, success=success)

        return ActionResult(
            success=success,
            output=self.decision_engine._format_result(tool_name, result),
            error=result.get("error"),
            tool_name=tool_name,
            duration_ms=elapsed,
            metadata={"raw_result": result},
        )

    # ══════════════════════════════════════════════════════════════════
    #  ACT — Path C: CREATE a new tool
    # ══════════════════════════════════════════════════════════════════

    async def _act_create(
        self, thought: Thought, user_input: str
    ) -> ActionResult:
        """Create a brand-new tool via the self-evolving brain.

        Steps:
            1. Plan tool spec via DecisionEngine
            2. Risk check — reject Level 3, prompt for Level 2
            3. Synthesize code with retry (max 3 attempts)
            4. Register in brain.db, save to generated/
            5. Execute the tool for real and return the result

        Args:
            thought: The reasoning output.
            user_input: Original user input.

        Returns:
            ActionResult from the newly created tool.
        """
        start = time.time()

        # ── Step 1: Plan tool creation ────────────────────────────────
        tool_spec = await self.decision_engine.plan_tool_creation(thought)
        if tool_spec is None:
            return ActionResult(
                success=False,
                error="Failed to plan tool specification",
                duration_ms=int((time.time() - start) * 1000),
            )

        # ── Step 2: Risk check ────────────────────────────────────────
        if tool_spec.estimated_risk >= 3:
            return ActionResult(
                success=False,
                error=(
                    f"Tool '{tool_spec.name}' requires dangerous operations "
                    f"(risk level {tool_spec.estimated_risk}) — blocked for safety."
                ),
                tool_name=tool_spec.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        if tool_spec.estimated_risk == 2:
            approved = await self.ui.prompt_tool_approval(tool_spec)
            if not approved:
                return ActionResult(
                    success=False,
                    error=f"Tool '{tool_spec.name}' creation denied by user (risk level 2).",
                    tool_name=tool_spec.name,
                    duration_ms=int((time.time() - start) * 1000),
                )

        # ── Step 3: Synthesize with retry ─────────────────────────────
        synth_result = await self.synthesizer.synthesize_with_retry(
            task=tool_spec.description,
            tool_name=tool_spec.name,
            inputs=tool_spec.inputs,
            outputs=tool_spec.outputs,
            safe_libraries=tool_spec.safe_libraries,
            max_attempts=self.config.get("brain", {}).get("max_synthesis_retries", 3),
        )

        if not synth_result.get("success", False):
            return ActionResult(
                success=False,
                error=synth_result.get("error", "Synthesis failed"),
                tool_name=tool_spec.name,
                duration_ms=int((time.time() - start) * 1000),
                metadata={"attempts": synth_result.get("attempts", 0)},
            )

        # ── Step 4: Tool is registered (synthesize_with_retry does it) ─
        registered_name = synth_result["name"]
        new_tool = self.registry.get_by_name(registered_name)
        if not new_tool:
            return ActionResult(
                success=False,
                error=f"Tool '{registered_name}' registered but not found in DB",
                tool_name=registered_name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Load the newly created tool
        self.loader.load(new_tool["name"], new_tool["code_path"])

        self._log_event("tool_created", registered_name, {
            "version": synth_result.get("version", 1),
            "code_path": synth_result.get("code_path", ""),
            "attempts": synth_result.get("attempts", 1),
        })

        # ── Step 5: Execute the tool for real ─────────────────────────
        func = self.loader.get(registered_name)
        if func is None:
            return ActionResult(
                success=False,
                error=f"Tool '{registered_name}' created but could not be loaded",
                tool_name=registered_name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Extract real parameters from user input
        params = await self.decision_engine._extract_params(user_input, new_tool)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: func(**params)
            )
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            logger.error("New tool %s execution error: %s", registered_name, exc)
            self.registry.update_stats(registered_name, success=False)
            return ActionResult(
                success=False,
                error=f"Tool created but execution failed: {exc}",
                tool_name=registered_name,
                duration_ms=elapsed,
            )

        elapsed = int((time.time() - start) * 1000)

        if not isinstance(result, dict) or "success" not in result:
            self.registry.update_stats(registered_name, success=False)
            return ActionResult(
                success=False,
                error="New tool returned invalid result format",
                tool_name=registered_name,
                duration_ms=elapsed,
            )

        success = result.get("success", False)
        self.registry.update_stats(registered_name, success=success)

        return ActionResult(
            success=success,
            output=self.decision_engine._format_result(registered_name, result),
            error=result.get("error"),
            tool_name=registered_name,
            duration_ms=elapsed,
            metadata={
                "raw_result": result,
                "code_path": synth_result.get("code_path", ""),
                "version": synth_result.get("version", 1),
                "newly_created": True,
            },
        )

    # ══════════════════════════════════════════════════════════════════
    #  OBSERVE — evaluate the action result
    # ══════════════════════════════════════════════════════════════════

    def _observe(self, thought: Thought, result: ActionResult) -> Observation:
        """Evaluate whether the action result fulfils the request.

        Args:
            thought: The reasoning that led to this action.
            result: The action result to evaluate.

        Returns:
            Observation with completeness and quality assessment.
        """
        if not result.success:
            return Observation(
                complete=False,
                quality=0.0,
                feedback=result.error or "Action failed with no error message",
            )

        if result.output is None or str(result.output).strip() == "":
            return Observation(
                complete=False,
                quality=0.1,
                feedback="Action succeeded but returned empty output",
            )

        # Check expected output fields if specified
        missing = []
        if thought.tool_outputs and isinstance(result.metadata.get("raw_result"), dict):
            raw = result.metadata["raw_result"].get("result", {})
            if isinstance(raw, dict):
                for field_name in thought.tool_outputs:
                    if field_name not in raw:
                        missing.append(field_name)

        if missing:
            return Observation(
                complete=False,
                quality=0.5,
                missing_fields=missing,
                feedback=f"Missing expected fields: {', '.join(missing)}",
            )

        # Calculate quality score
        output_str = str(result.output)
        quality = 1.0
        if len(output_str) < 10:
            quality = 0.6
        if "error" in output_str.lower() and "no error" not in output_str.lower():
            quality *= 0.7

        return Observation(
            complete=True,
            quality=quality,
            feedback="Result looks complete",
        )

    # ══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _log_event(self, event_type: str, name: str, details: dict) -> None:
        """Log a react engine event."""
        try:
            if hasattr(self.interaction_logger, "log_tool_event"):
                self.interaction_logger.log_tool_event(event_type, name, details)
        except Exception as exc:
            logger.debug("Failed to log react event: %s", exc)
