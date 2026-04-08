"""Orchestrator — Central async coordinator for JARVIS."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from core.llm_core import TaskType

logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS", version="1.0.0")


class TextInput(BaseModel):
    """Input model for the /input endpoint."""
    text: str


class MemoryInput(BaseModel):
    """Input model for the /memory/add endpoint."""
    text: str
    category: str = "fact"


class Orchestrator:
    """The central nervous system of JARVIS.

    Coordinates all modules, exposes a FastAPI server, and implements
    the full perception-reasoning-action loop.
    """

    def __init__(
        self,
        config: dict,
        intent_classifier,
        context_builder,
        llm_core,
        os_agent,
        file_agent,
        image_analyzer,
        terminal_agent,
        memory_system,
        personal_model,
        interaction_logger,
        response_cache,
        ui_layer,
        web_agent=None,
        checkpoint_manager=None,
        data_agent=None,
        email_agent=None,
        scheduler_agent=None,
        api_agent=None,
        workflow_agent=None,
    ) -> None:
        """Initialize the orchestrator with all module references.

        Args:
            config: Full JARVIS configuration dictionary.
            intent_classifier: IntentClassifier instance.
            context_builder: ContextBuilder instance.
            llm_core: LLMCore instance.
            os_agent: OSAgent instance.
            file_agent: FileAgent instance.
            image_analyzer: ImageAnalyzer instance.
            terminal_agent: TerminalAgent instance.
            memory_system: MemorySystem instance.
            personal_model: PersonalModel instance.
            interaction_logger: InteractionLogger instance.
            response_cache: ResponseCache instance.
            ui_layer: UILayer instance.
            web_agent: Optional WebAgent instance for web search/fetch.
            checkpoint_manager: Optional CheckpointManager for undo support.
        """
        self.config = config
        self.classifier = intent_classifier
        self.context_builder = context_builder
        self.llm = llm_core
        self.os_agent = os_agent
        self.file_agent = file_agent
        self.image_analyzer = image_analyzer
        self.terminal_agent = terminal_agent
        self.memory = memory_system
        self.personal_model = personal_model
        self.interaction_logger = interaction_logger
        self.cache = response_cache
        self.ui = ui_layer
        self.web_agent = web_agent
        self.checkpoint = checkpoint_manager
        self.data_agent = data_agent
        self.email_agent = email_agent
        self.scheduler_agent = scheduler_agent
        self.api_agent = api_agent
        self.workflow_agent = workflow_agent
        self.decision_engine = None  # Brain Decision Engine — set after init if available
        self.mcp_client = None  # MCP Client — set after init if available
        self.executor = ThreadPoolExecutor(
            max_workers=config["orchestrator"]["thread_pool_workers"]
        )
        # Conversation state
        self._last_action: dict | None = None  # {"intent": str, "input": str, "response": str}
        self._media_brain_pending: bool = False  # Flag for media queries to use lower brain threshold
        self.react_engine = None  # ReActEngine — set after init if available
        self._conversation_history: list[dict] = []  # [{"user": str, "jarvis": str}, ...]
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register FastAPI endpoints."""

        @app.post("/input")
        async def handle_text_input(body: TextInput) -> dict:
            response = await self.handle_input(body.text)
            return {"response": response}

        @app.get("/status")
        async def get_status() -> dict:
            return self.os_agent.get_system_state()

        @app.get("/health")
        async def health_check() -> dict:
            return {"status": "ok"}

        @app.post("/memory/add")
        async def add_memory(body: MemoryInput) -> dict:
            self.memory.add_memory(body.text, body.category)
            return {"status": "added"}

        @app.get("/memory/query")
        async def query_memory(q: str = "") -> dict:
            results = self.memory.query(q, top_k=5)
            return {"results": results}

        @app.post("/undo")
        async def undo() -> dict:
            if self.checkpoint:
                result = self.checkpoint.undo()
                return {"result": result}
            return {"result": "Checkpoint manager not available."}

        @app.get("/checkpoints")
        async def list_checkpoints() -> dict:
            if self.checkpoint:
                return {"checkpoints": self.checkpoint.list_checkpoints()}
            return {"checkpoints": "Checkpoint manager not available."}

    # Phrases that mean "do the same thing again"
    _REPEAT_TRIGGERS = {
        "do it once", "do it again", "do it one more time", "again", "repeat",
        "repeat that", "repeat it", "one more time", "same thing", "do the same",
        "do that again", "do it", "run it again", "try again", "redo",
        "redo that", "redo it", "yes do it", "yes, do it", "go ahead",
        "do it now", "just do it", "execute again",
    }

    def _is_repeat_command(self, text: str) -> bool:
        """Return True if the text is a short repeat/confirmation phrase."""
        lower = text.strip().lower().rstrip("!.?")
        return lower in self._REPEAT_TRIGGERS

    def _record_turn(self, user_input: str, response: str, intent: str) -> None:
        """Save this turn to history and update last_action."""
        self._last_action = {"intent": intent, "input": user_input, "response": response}
        self._conversation_history.append({"user": user_input, "jarvis": response})
        # Keep only the last 10 turns
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]

    def _build_history_snippet(self) -> str:
        """Return recent conversation as a compact string for LLM context injection."""
        if not self._conversation_history:
            return ""
        lines: list[str] = []
        for turn in self._conversation_history[-4:]:  # last 4 turns max
            lines.append(f"User: {turn['user']}")
            lines.append(f"JARVIS: {turn['jarvis'][:200]}")
        return "\n".join(lines)

    async def _extract_mcp_arguments(self, user_input: str, tool_match: dict) -> dict[str, Any]:
        """Use the fast LLM to map a user request to MCP tool arguments."""
        low = user_input.lower().strip()
        tool_name = tool_match.get("name", "")

        if tool_name == "windows_mcp__PowerShell":
            command = user_input
            for prefix in (
                "run powershell command",
                "execute powershell command",
                "powershell command",
                "run powershell",
                "execute powershell",
            ):
                if low.startswith(prefix):
                    command = user_input[len(prefix):].strip(" :")
                    break
            if command and command != user_input:
                return {"command": command, "timeout": 20}

        if tool_name == "windows_mcp__Process":
            args: dict[str, Any] = {
                "mode": "kill" if any(w in low for w in ("kill", "terminate", "stop process")) else "list"
            }
            if "cpu" in low:
                args["sort_by"] = "cpu"
            elif "name" in low:
                args["sort_by"] = "name"
            else:
                args["sort_by"] = "memory"
            m = re.search(r"\b(\d+)\b", low)
            if m and args["mode"] == "list":
                args["limit"] = int(m.group(1))
            if args["mode"] == "kill":
                pid_match = re.search(r"\bpid\s+(\d+)\b", low)
                if pid_match:
                    args["pid"] = int(pid_match.group(1))
                else:
                    name_match = re.search(r"(?:kill|terminate|stop process)\s+([a-zA-Z0-9_.-]+)", user_input, re.I)
                    if name_match:
                        args["name"] = name_match.group(1)
                        args["force"] = True
            return args

        if tool_name == "windows_mcp__Clipboard":
            if any(w in low for w in ("what is in my clipboard", "show clipboard", "read clipboard", "clipboard content")):
                return {"mode": "get"}
            text_match = re.search(
                r"(?:copy|set)\s+(.+?)\s+(?:to|into)\s+clipboard\b|clipboard\s+(?:to|as)\s+(.+)$",
                user_input,
                re.I,
            )
            if text_match:
                text = (text_match.group(1) or text_match.group(2) or "").strip(" .\"'")
                if text:
                    return {"mode": "set", "text": text}
            if "clipboard" in low:
                return {"mode": "get"}

        if tool_name == "windows_mcp__Notification":
            message = user_input
            for pattern in (
                r"(?:notification saying|notify me saying|send a windows notification saying)\s+(.+)$",
                r"(?:notification with message|notify me with message)\s+(.+)$",
            ):
                m = re.search(pattern, user_input, re.I)
                if m:
                    message = m.group(1).strip()
                    break
            return {"title": "JARVIS", "message": message.strip(), "app_id": "JARVIS"}

        if tool_name == "windows_mcp__Snapshot":
            return {"use_annotation": True, "use_ui_tree": True}

        if tool_name == "windows_mcp__Screenshot":
            return {"use_annotation": False}

        if tool_name == "windows_mcp__App":
            args = {"mode": "switch" if any(w in low for w in ("switch", "focus")) else "launch"}
            app_match = re.search(
                r"(?:open|launch|start|switch to|focus|focus on)\s+(.+?)(?:\s+window)?$",
                user_input,
                re.I,
            )
            if app_match:
                args["name"] = app_match.group(1).strip()
            return args

        schema = tool_match.get("inputSchema", {}) or {}
        properties = schema.get("properties", {}) or {}
        if not properties:
            return {}

        prompt = (
            f"User request: {user_input}\n"
            f"Tool: {tool_match.get('name', '')}\n"
            f"Description: {tool_match.get('description', '')}\n"
            f"Schema: {json.dumps(schema, ensure_ascii=True)}\n\n"
            "Return only a JSON object of tool arguments.\n"
            "Use obvious defaults when the request implies them.\n"
            "Omit unknown optional fields.\n"
            "If nothing is inferable, return {}."
        )

        try:
            raw = await self.llm.generate(
                prompt,
                task_type=TaskType.INTENT_CLASSIFY,
                system_prompt="Extract tool-call arguments. Return JSON only.",
            )
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end < start:
                return {}
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug("MCP argument extraction failed: %s", exc)
            return {}

    async def handle_input(self, user_input: str) -> str:
        """Process a user input through the full JARVIS pipeline.

        Implements the request flow: parallel preprocessing -> route -> execute -> learn.

        Args:
            user_input: The user's text input.

        Returns:
            JARVIS response string.
        """
        start_time = time.time()
        self.ui.display_user_input(user_input)

        # Handle undo/checkpoint commands immediately
        lower_input = user_input.strip().lower()
        if lower_input in ("undo", "undo last", "revert", "revert last"):
            if self.checkpoint:
                result = self.checkpoint.undo()
                self.ui.display_response(result, intent="UNDO")
                return result
            return "Checkpoint manager not available."
        if lower_input in ("checkpoints", "show checkpoints", "list checkpoints"):
            if self.checkpoint:
                result = self.checkpoint.list_checkpoints()
                self.ui.display_response(result, intent="CHECKPOINT_LIST")
                return result
            return "No checkpoint manager available."

        # On-demand memory audit
        if any(
            phrase in lower_input
            for phrase in ("clean your memory", "clean memory", "memory audit", "audit memory")
        ):
            if hasattr(self, "memory_auditor") and self.memory_auditor:
                self.ui.console.print(
                    "  [dim cyan]>> Running on-demand memory audit...[/dim cyan]"
                )
                report = await self.memory_auditor.run_audit()
                self.ui.show_audit_report(report)
                result = f"Memory audit complete: {report.summary}"
                self.ui.display_response(result, intent="SYSTEM_QUERY")
                return result
            return "Memory auditor not available."

        # Source attribution follow-up — return source directly, no LLM hallucination
        _SOURCE_PHRASES = (
            "where did you get", "where you got", "what is the source",
            "what source", "how do you know that", "where are you getting",
            "where did that come from", "where does that come from",
            "where you get", "source of that", "data source",
        )
        if any(phrase in lower_input for phrase in _SOURCE_PHRASES):
            if self._last_action and self._last_action.get("intent") == "API_TASK":
                response_text = self._last_action.get("response", "")
                import re as _re
                source_match = _re.search(r'📡 Source:[^\n]+', response_text)
                if source_match:
                    reply = (
                        f"I fetched that from a live real-time API.\n"
                        f"{source_match.group(0)}"
                    )
                    self._record_turn(user_input, reply, "SYSTEM_QUERY")
                    self.ui.display_response(reply, intent="SYSTEM_QUERY")
                    self.ui.speak(reply[:200])
                    return reply

        # On-demand session learning summary
        if any(
            phrase in lower_input
            for phrase in (
                "what did you learn", "what have you learned",
                "lessons learned", "session lessons",
            )
        ):
            if hasattr(self, "session_learner") and self.session_learner:
                result = await self.session_learner.get_lessons_learned()
                self.ui.display_response(result, intent="SYSTEM_QUERY")
                return result
            return "Session learner not available."

        # Identity question — "who am I", "what is my name", "do you know me"
        _IDENTITY_PHRASES = (
            "who am i", "what is my name", "what's my name", "do you know me",
            "do you know who i am", "who is your boss", "who is your owner",
        )
        if any(phrase in lower_input for phrase in _IDENTITY_PHRASES):
            reply = (
                "You are Faheem, my Boss. "
                "I am JARVIS, your personal AI assistant."
            )
            self._record_turn(user_input, reply, "CHAT")
            self.ui.display_response(reply, intent="CHAT")
            self.ui.speak(reply)
            return reply

        # Workflow auto-detection — before classification
        if self.workflow_agent:
            from agents.workflow_agent import WorkflowAgent
            if WorkflowAgent.is_workflow(user_input):
                self.ui.console.print("  [dim magenta]>> Detected multi-step workflow[/dim magenta]")
                context = await self.context_builder.build_context(user_input, mode="code")
                response = await self.workflow_agent.handle(user_input, context)
                self._record_turn(user_input, response, "WORKFLOW")
                self.ui.display_response(response, intent="WORKFLOW")
                return response

        # Repeat / "do it again" detection — replay last action
        if self._is_repeat_command(user_input):
            if self._last_action:
                self.ui.console.print(
                    f"  [dim cyan]>> Repeating last action: [{self._last_action['intent']}] "
                    f"{self._last_action['input'][:60]}[/dim cyan]"
                )
                return await self.handle_input(self._last_action["input"])
            else:
                reply = "Nothing to repeat yet — I haven't done anything this session."
                self.ui.display_response(reply, intent="CHAT")
                return reply

        # Classify first so we know whether to use cache (code/terminal results must not be cached)
        loop = asyncio.get_event_loop()

        # ── Hard override — GUI automation patterns always → OS_COMMAND ────
        # Runs BEFORE the sentence transformer classifier so pattern-matched
        # GUI tasks never get misrouted to WEB_TASK or CHAT.
        _GUI_PATTERNS = [
            r'type\s+.+\s+in\s+\w+',
            r'write\s+.+\s+in\s+\w+',
            r'open\s+\w+\s+and\s+(?:type|write)',
            r'launch\s+\w+\s+and\s+(?:type|write)',
            r'in\s+\w+\s+(?:type|write)\s+',
            r'click\s+.+\s+in\s+\w+',
            r'press\s+.+\s+in\s+\w+',
            r'minimi[sz]e\s+all\s+(?:windows|screens?)',
            r'show\s+desktop',
        ]
        gui_override = False
        for pattern in _GUI_PATTERNS:
            if re.search(pattern, user_input, re.I):
                gui_override = True
                break

        if gui_override:
            intent_result_pre = ("OS_COMMAND", 1.0)
            logger.info("GUI pattern override → OS_COMMAND (skipped classifier)")
        else:
            intent_result_pre = await loop.run_in_executor(
                self.executor, self.classifier.classify, user_input
            )
        # intent_result_pre is (intent_str, confidence_float) — keep as tuple
        intent_pre = intent_result_pre[0]

        # Skip cache for dynamic intents — code/terminal/web results change with file content
        _NO_CACHE_INTENTS = {"CODE_TASK", "TERMINAL_TASK", "WEB_TASK", "IMAGE_ANALYZE", "SYSTEM_QUERY", "API_TASK"}
        use_cache = intent_pre not in _NO_CACHE_INTENTS
        # Also skip cache when input contains a URL or domain (likely web fetch/check)
        if use_cache and re.search(r"https?://\S+|(?:www\.)?[\w-]+\.(?:com|org|net|io|dev|ai|co|gov|edu)", user_input.lower()):
            use_cache = False

        if use_cache:
            cached = self.cache.get(user_input)
            if cached:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._record_turn(user_input, cached, intent_pre)
                self.ui.display_response(cached, intent="CACHE_HIT")
                self.ui.speak(cached)
                self.interaction_logger.log(user_input, cached, "CACHE_HIT", elapsed_ms, cache_hit=True)
                return cached

        try:
            # Try parallel task splitting first ("do X and Y" with different intents)
            parallel_result = await self._try_parallel_tasks(user_input)
            if parallel_result is not None:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._record_turn(user_input, parallel_result, "PARALLEL")
                self.ui.display_response(parallel_result, intent="PARALLEL")
                self.ui.speak(parallel_result[:300])
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._learn, user_input, parallel_result, "PARALLEL", elapsed_ms
                )
                return parallel_result

            # Try sequential multi-step ("do X then Y")
            sequential_result = await self._try_sequential_steps(user_input)
            if sequential_result is not None:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._record_turn(user_input, sequential_result, "SEQUENTIAL")
                self.ui.display_response(sequential_result, intent="SEQUENTIAL")
                self.ui.speak(sequential_result[:300])
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._learn, user_input, sequential_result, "SEQUENTIAL", elapsed_ms
                )
                return sequential_result

            # Build context (reuse the intent we already classified above)
            intent, confidence = intent_result_pre
            context = await self.context_builder.build_context(
                user_input, mode=self._intent_to_mode(intent)
            )

            # ── Real-time data override ───────────────────────────────────
            # These keywords MUST go to api_agent regardless of classifier.
            # The intent classifier often sends them to WEB_TASK instead.
            low_in = user_input.lower()
            _API_KEYWORDS = (
                "weather", "temperature", "forecast", "rain", "humidity",
                "bitcoin", "crypto", "ethereum", "btc", "eth", "coin price",
                "exchange rate", "usd to", "inr to", "eur to", "convert currency",
                "my ip", "public ip",
                "tell me a joke", "random joke",
                "top news", "latest news", "news headlines",
            )
            if any(kw in low_in for kw in _API_KEYWORDS) and self.api_agent:
                if intent != "API_TASK":
                    logger.info(
                        "Real-time keyword override: %s (%.2f) → API_TASK", intent, confidence
                    )
                    intent = "API_TASK"

            # ── Web URL / fetch / site-status override ────────────────────
            # If input contains a URL or domain, route to WEB_TASK so web_agent
            # can fetch/check it instead of the LLM hallucinating.
            if intent not in ("WEB_TASK", "EMAIL_TASK") and self.web_agent:
                _has_url = bool(re.search(r"https?://\S+", user_input))
                _has_domain = bool(re.search(
                    r"\b(?:www\.)?[\w-]+\.(?:com|org|net|io|dev|ai|co|gov|edu)(?:/\S*)?\b", low_in
                ))
                _web_targets = ("wikipedia", "reddit", "stackoverflow", "github")
                _has_web_target = any(t in low_in for t in _web_targets)
                _web_verbs = ("fetch", "scrape", "crawl", "is ", "check if",
                              "site status", "search ", "content of", "open ")
                _has_web_verb = any(v in low_in for v in _web_verbs)
                if _has_url or ((_has_domain or _has_web_target) and _has_web_verb):
                    logger.info(
                        "Web URL/verb override: %s (%.2f) → WEB_TASK", intent, confidence
                    )
                    intent = "WEB_TASK"

            # Low-confidence CHAT — try to rescue with better routing
            if intent == "CHAT" and confidence < 0.55:

                # Follow-up to previous IMAGE_ANALYZE — re-ask about the same screenshot
                _image_follow = ("what is that", "which software", "which app", "what app",
                                 "what software", "read the text", "what does it say",
                                 "tell me more", "identify", "what error", "is that",
                                 "can you find", "what's on", "describe")
                if (self._last_action and self._last_action["intent"] == "IMAGE_ANALYZE"
                        and any(s in low_in for s in _image_follow)):
                    intent = "IMAGE_ANALYZE"
                    logger.info("Low-confidence CHAT (%.2f) + IMAGE_ANALYZE follow-up → rerouting to IMAGE_ANALYZE", confidence)

                # Media / music playback — try brain tool first (music_controller)
                elif any(s in low_in for s in ("play ", "song", "music", "pause",
                                                "next song", "previous song", "stop music", "resume")):
                    self._media_brain_pending = True
                    intent = "OS_COMMAND"
                    logger.info("Low-confidence CHAT (%.2f) + media signals → rerouting to OS_COMMAND (brain priority)", confidence)

                # Desktop/window OS actions
                elif any(s in low_in for s in ("minimize all", "show desktop",
                                                "hide all windows", "clear desktop",
                                                "open ", "launch ", "start ",
                                                "close all windows", "mute", "unmute",
                                                "volume", "screenshot")):
                    intent = "OS_COMMAND"
                    logger.info("Low-confidence CHAT (%.2f) + OS signals → rerouting to OS_COMMAND", confidence)

                # File operations — require stronger signals to avoid false positives
                elif any(s in low_in for s in ("my pic", "my photo", "my image", "my video",
                                                "my file", "my doc", "on e drive", "on d drive",
                                                "e drive", "d drive", "downloads folder",
                                                "desktop folder", "documents folder",
                                                "where are my", "find my", "locate my",
                                                "show my files", "show my pic", "show my photo",
                                                "create folder", "make folder", "create directory",
                                                "move folder", "rename folder",
                                                "where is", "locate", "find file", "java file",
                                                "document file", "folder file")):
                    intent = "FILE_OP"
                    logger.info("Low-confidence CHAT (%.2f) + file signals → rerouting to FILE_OP", confidence)

                # System queries
                elif any(s in low_in for s in ("cpu", "ram", "memory usage", "disk space",
                                                "gpu", "usage", "battery", "vram",
                                                "wifi", "signal strength", "network",
                                                "uptime", "temperature", "screen resolution",
                                                "count", "lines of code", "python files",
                                                "how many files", "largest file",
                                                "startup", "start with windows", "autostart",
                                                "started with windows", "boot with",
                                                "running processes", "installed programs")):
                    intent = "SYSTEM_QUERY"
                    logger.info("Low-confidence CHAT (%.2f) + sys signals → rerouting to SYSTEM_QUERY", confidence)

                elif self._last_action and len(user_input.split()) <= 5 and not any(
                    s in low_in for s in ("where is", "locate", "find ", "file", "folder", "document")
                ):
                    # Very short + prior context → keep as CHAT follow-up
                    logger.info("Low-confidence CHAT (%.2f) + short + history → keeping CHAT", confidence)

                # Email signals
                elif any(s in low_in for s in ("email", "send mail", "inbox", "send report",
                                                "email me", "email attachment", "notify me",
                                                "forward", "compose mail")):
                    intent = "EMAIL_TASK"
                    logger.info("Low-confidence CHAT (%.2f) + email signals → rerouting to EMAIL_TASK", confidence)

                # File save/create signals
                elif any(s in low_in for s in ("save to", "save as", "create document",
                                                "new document", "write to file", "save file",
                                                "save report", "copy and paste", "paste into",
                                                "save result", "rename file", "move file",
                                                "word document", "save summarized", "save invoices")):
                    intent = "FILE_OP"
                    logger.info("Low-confidence CHAT (%.2f) + file-save signals → rerouting to FILE_OP", confidence)

                # Data analysis signals
                elif any(s in low_in for s in ("csv", "excel", "chart", "plot", "histogram",
                                                "spreadsheet", "analyze data", "dataset")):
                    intent = "DATA_TASK"
                    logger.info("Low-confidence CHAT (%.2f) + data signals → rerouting to DATA_TASK", confidence)

                # Schedule signals
                elif any(s in low_in for s in ("remind me", "schedule", "every hour",
                                                "every day", "at 3pm", "at 9am")):
                    intent = "SCHEDULE_TASK"
                    logger.info("Low-confidence CHAT (%.2f) + schedule signals → rerouting to SCHEDULE_TASK", confidence)

                elif self.web_agent:
                    intent = "WEB_TASK"
                    logger.info("Low-confidence CHAT (%.2f) → rerouting to WEB_TASK", confidence)

            logger.info("Intent: %s (%.2f)", intent, confidence)

            # ── ReAct loop — runs BEFORE route_to_agent to prevent re-entry ──
            history_snippet = self._build_history_snippet()
            react_enabled = self.config.get("react", {}).get("enabled", False)
            if react_enabled and self.react_engine is not None:
                if intent in ("API_TASK",) or self._is_brain_request(user_input):
                    try:
                        react_response = await self.react_engine.run(user_input, context)
                        if react_response:
                            response = react_response
                            elapsed_ms = int((time.time() - start_time) * 1000)
                            self._record_turn(user_input, response, intent)
                            self.ui.display_response(response, intent=intent)
                            self.ui.speak(response)
                            return response
                    except Exception as react_exc:
                        logger.debug("ReAct engine error, falling back: %s", react_exc)

            # Route to agent
            response = await self.route_to_agent(intent, user_input, context, history_snippet)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Record turn for future repeat/context use
            self._record_turn(user_input, response, intent)

            # Display and speak
            self.ui.display_response(response, intent=intent)
            self.ui.speak(response)

            # Only cache non-dynamic responses
            if use_cache:
                self.cache.set(user_input, response)

            # Learning (async — does not delay response)
            asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._learn,
                user_input,
                response,
                intent,
                elapsed_ms,
            )

            return response

        except Exception as exc:
            logger.error("Error handling input: %s", exc)
            error_msg = f"An error occurred: {exc}"
            self.ui.show_error(error_msg)
            return error_msg

    # File extensions to look for in CODE_TASK inputs
    _CODE_EXT_PATTERN = re.compile(
        r'[\w/\\.\-]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|cs|rb|php|'
        r'sh|bat|ps1|yaml|yml|json|toml|ini|cfg|md|txt|html|css|vue|svelte)',
        re.IGNORECASE,
    )

    def _extract_file_refs(self, user_input: str) -> list[str]:
        """Extract code file paths mentioned in user input.

        Args:
            user_input: The user's text.

        Returns:
            List of unique file path strings found.
        """
        return list(dict.fromkeys(self._CODE_EXT_PATTERN.findall(user_input)))

    async def _handle_code_task(self, user_input: str, context: dict) -> str:
        """Handle CODE_TASK: read referenced files, then call LLM with full context.

        This is the core of Claude Code-style behaviour — never answer blind,
        always read the file first.

        Args:
            user_input: The user's code-related request.
            context: Context dict from ContextBuilder.

        Returns:
            LLM response string with code/explanation.
        """
        system_prompt = context.get("system_prompt", "")
        file_refs = self._extract_file_refs(user_input)
        file_context = ""

        for ref in file_refs[:3]:  # Read up to 3 files
            content = None
            try:
                content = self.file_agent.read_file(ref)
            except FileNotFoundError:
                # Try searching for the file by name
                fname = ref.replace("\\", "/").split("/")[-1]
                results = self.file_agent.search_by_name(fname)
                if results:
                    try:
                        content = self.file_agent.read_file(str(results[0]))
                        ref = str(results[0])  # Use resolved path in output
                    except Exception:
                        pass
            except Exception:
                pass

            if content:
                # Limit each file to 2500 chars to stay within context window
                snippet = content[:2500] + ("\n... (truncated)" if len(content) > 2500 else "")
                file_context += f"\n\n```\n# File: {ref}\n{snippet}\n```"
                self.ui.console.print(
                    f"  [dim cyan]>> Read file: {ref} ({len(content)} chars)[/dim cyan]"
                )

        if file_context:
            prompt = f"Task: {user_input}\n\nRelevant files:{file_context}"
        else:
            prompt = user_input

        return await self.llm.generate(
            prompt,
            task_type=TaskType.CODE_WRITE,
            system_prompt=system_prompt
        )

    @staticmethod
    def _intent_to_mode(intent: str) -> str:
        """Map an intent label to an LLM mode string."""
        return {
            "CODE_TASK": "code",
            "TERMINAL_TASK": "code",
            "SYSTEM_QUERY": "system",
            "WEB_TASK": "web",
            "API_TASK": "web",
            "DATA_TASK": "code",
            "EMAIL_TASK": "code",
            "SCHEDULE_TASK": "chat",
            "WORKFLOW": "code",
        }.get(intent, "chat")

    async def run_parallel_preprocessing(
        self, user_input: str
    ) -> tuple[tuple[str, float], dict]:
        """Run intent classification and context building in parallel.

        Args:
            user_input: The user's text input.

        Returns:
            Tuple of (intent_result, context_dict).
        """
        loop = asyncio.get_event_loop()
        intent_future = loop.run_in_executor(
            self.executor, self.classifier.classify, user_input
        )
        intent_result = await intent_future
        intent, _ = intent_result
        context = await self.context_builder.build_context(
            user_input, mode=self._intent_to_mode(intent)
        )
        return intent_result, context

    async def route_to_agent(
        self, intent: str, user_input: str, context: dict, history_snippet: str = ""
    ) -> str:
        """Route the request to the appropriate agent based on intent.

        Args:
            intent: Classified intent label.
            user_input: The user's text input.
            context: Assembled context dict from ContextBuilder.
            history_snippet: Recent conversation history for LLM context injection.

        Returns:
            Response string from the routed agent.
        """
        system_prompt = context.get("system_prompt", "")
        # Prepend conversation history to system prompt for context-aware replies
        if history_snippet and intent in ("CHAT", "CODE_TASK", "MEMORY_QUERY"):
            system_prompt = (
                f"Recent conversation:\n{history_snippet}\n\n---\n{system_prompt}"
            )

        # NOTE: ReAct loop is handled in handle_input BEFORE route_to_agent
        # to prevent infinite recursion (ReAct → _act_agent → route_to_agent → ReAct).
        # Do NOT add a ReAct block here.

        # Brain intercept — if user explicitly asks to "create a tool",
        # skip reasoning and go straight to tool synthesis
        if self.decision_engine and self._is_brain_request(user_input):
            try:
                brain_result = await self.decision_engine.handle(
                    user_input, context, force_create=True
                )
                if brain_result is not None:
                    return brain_result
            except Exception as brain_exc:
                logger.debug("Brain intercept error: %s", brain_exc)

        # ── Brain existing tool check — Priority 0.5 ─────────────────────
        # Before routing to standard agents, check if the brain already has
        # a specialized tool for this query (e.g. gpu_temperature, uptime).
        if self.decision_engine:
            try:
                # Use lower threshold for media queries (short commands like
                # "next song" have low embedding similarity with descriptions)
                media_pending = getattr(self, '_media_brain_pending', False)
                self._media_brain_pending = False
                low_in = user_input.lower()
                _media_signals = ("play ", "song", "music", "pause", "next song",
                                  "previous song", "stop music", "resume",
                                  "skip song", "now playing", "what's playing",
                                  "currently playing")
                is_media = media_pending or any(s in low_in for s in _media_signals)
                thresh = 0.25 if is_media else None
                brain_result = await self.decision_engine.try_existing_tool(
                    user_input, threshold=thresh
                )
                if brain_result is not None:
                    return brain_result
            except Exception as brain_exc:
                logger.debug("Brain existing tool check error: %s", brain_exc)

        # ── Brain full reasoning — for queries standard agents can't handle ──
        # Let the brain reason about whether to CREATE a new tool or USE an agent.
        # Skip for intents with dedicated reliable handlers (file, terminal, image, etc.)
        _BRAIN_SKIP_INTENTS = {"FILE_OP", "TERMINAL_TASK", "IMAGE_ANALYZE",
                               "EMAIL_TASK", "SCHEDULE_TASK", "DATA_TASK", "WORKFLOW",
                               "OS_COMMAND"}  # os_agent handles these directly
        if self.decision_engine and intent not in _BRAIN_SKIP_INTENTS:
            try:
                brain_result = await self.decision_engine.handle(user_input, context)
                if brain_result is not None:
                    return brain_result
                # If brain recommended an agent, re-route to it
                if self.decision_engine.recommended_agent:
                    rec = self.decision_engine.recommended_agent
                    self.decision_engine.recommended_agent = None
                    rec_intent = self._agent_to_intent(rec)
                    if rec_intent and rec_intent != intent:
                        logger.info("Brain recommended %s → re-routing to %s", rec, rec_intent)
                        intent = rec_intent
            except Exception as brain_exc:
                logger.debug("Brain reasoning error: %s", brain_exc)

        # ── MCP tool matching — Priority 1 ────────────────────────────────
        # Check MCP servers for a matching tool before falling through to
        # intent-based routing. This gives JARVIS access to hundreds of
        # production-ready tools from community MCP servers.
        # Skip MCP for intents that already have stronger native handlers.
        # Windows-MCP is most useful for desktop/system actions; broad matching on
        # file/workflow requests causes bad tool picks like App/Move/Snapshot.
        mcp_low = user_input.lower()
        mcp_allowed_intents = {"OS_COMMAND", "SYSTEM_QUERY"}
        mcp_block_terms = (
            "file", "folder", "directory", "desktop", "downloads", "documents",
            "report", "document", "pdf", "docx", "pptx", "html", "csv", "json",
            "save to", "save as", "create folder", "make folder", "move file",
            "move folder", "rename file", "rename folder",
            "screenshot", "screen shot", "capture screen", "snapshot",
        )
        allow_mcp = (
            self.mcp_client
            and self.mcp_client.enabled
            and intent in mcp_allowed_intents
            and not any(term in mcp_low for term in mcp_block_terms)
        )
        if allow_mcp:
            try:
                match = self.mcp_client.find_tool(user_input)
                if match and match.get("match_score", 0) >= 0.65:
                    tool_name = match["name"]
                    tool_args = await self._extract_mcp_arguments(user_input, match)
                    logger.info(
                        "MCP tool match: %s (score=%.2f, args=%s)",
                        tool_name, match["match_score"], tool_args,
                    )
                    self.ui.console.print(
                        f"  [dim cyan]>> MCP tool: {tool_name}[/dim cyan]"
                    )
                    mcp_result = await self.mcp_client.call_tool(tool_name, tool_args)
                    if mcp_result.get("success"):
                        result_text = str(mcp_result["result"])
                        if (
                            tool_name == "windows_mcp__App"
                            and tool_args.get("mode") == "switch"
                            and "no windows found" in result_text.lower()
                            and tool_args.get("name")
                        ):
                            fallback_name = str(tool_args["name"])
                            logger.info(
                                "MCP app switch found no windows; falling back to launch for %s",
                                fallback_name,
                            )
                            launch_result = await self.mcp_client.call_tool(
                                tool_name,
                                {"mode": "launch", "name": fallback_name},
                            )
                            if launch_result.get("success"):
                                result_text = str(launch_result["result"])
                        # Format through LLM for natural response
                        formatted = await self.llm.generate(
                            f"User asked: {user_input}\n\nTool result:\n{result_text[:2000]}\n\n"
                            "Summarize this result naturally for the user.",
                            task_type=TaskType.FORMAT_RESULT,
                            system_prompt=system_prompt
                        )
                        return formatted
            except Exception as mcp_exc:
                logger.debug("MCP tool match error: %s", mcp_exc)

        try:
            if intent == "TERMINAL_TASK":
                return await self.terminal_agent.execute(user_input, context)
            elif intent == "OS_COMMAND":
                response = await self.os_agent.execute(user_input, context)
                # If os_agent couldn't handle it, let the LLM agent loop try
                if "couldn't determine the specific action" in response:
                    return await self.terminal_agent.execute(user_input, context)
                return response
            elif intent == "FILE_OP":
                return await self.file_agent.handle(user_input, context)
            elif intent == "IMAGE_ANALYZE":
                # Follow-up: reuse last screenshot if this is a follow-up question
                if (self._last_action and self._last_action["intent"] == "IMAGE_ANALYZE"
                        and hasattr(self.image_analyzer, "last_screenshot")):
                    last_ss = self.image_analyzer.last_screenshot
                    if last_ss and Path(last_ss).exists():
                        logger.info("IMAGE_ANALYZE follow-up — reusing last screenshot: %s", last_ss)
                        return await self.image_analyzer.analyze_file(
                            last_ss, self.image_analyzer.build_prompt(user_input)
                        )
                return await self.image_analyzer.analyze_screenshot(
                    self.image_analyzer.build_prompt(user_input)
                )
            elif intent == "MEMORY_QUERY":
                results = self.memory.query(user_input, top_k=5)
                if results:
                    memory_text = "\n".join(
                        f"- {r['text'][:200]}" for r in results
                    )
                    prompt = f"Based on these memories:\n{memory_text}\n\nAnswer: {user_input}"
                    return await self.llm.generate(
                        prompt,
                        task_type=TaskType.MEMORY_SYNTHESIS,
                        system_prompt=system_prompt
                    )
                return await self.llm.generate(
                    user_input,
                    task_type=TaskType.SIMPLE_CHAT,
                    system_prompt=system_prompt
                )
            elif intent == "CODE_TASK":
                return await self._handle_code_task(user_input, context)
            elif intent == "SYSTEM_QUERY":
                import psutil as _psutil
                import re as _re
                state = self.os_agent.get_system_state()
                # Battery info
                battery_line = ""
                try:
                    batt = _psutil.sensors_battery()
                    if batt:
                        battery_line = f"  Battery: {batt.percent}% ({'charging' if batt.power_plugged else 'on battery'})\n"
                except Exception:
                    pass
                # Gather disk info for all valid drives (or just the one asked about)
                disk_lines = ""
                _low = user_input.lower()
                _drive_match = _re.search(r'\b([a-z])\s*drive\b', _low)
                if _drive_match:
                    _drives = [_drive_match.group(1).upper()]
                else:
                    _drives = ["C"]
                for _dl in _drives:
                    _dp = f"{_dl}:\\"
                    try:
                        _du = _psutil.disk_usage(_dp)
                        disk_lines += f"  Disk {_dl}: total: {round(_du.total/1024**3,1)} GB, used: {round(_du.used/1024**3,1)} GB, free: {round(_du.free/1024**3,1)} GB ({_du.percent}% used)\n"
                    except Exception:
                        disk_lines += f"  Disk {_dl}: not available\n"
                prompt = (
                    f"LIVE SYSTEM DATA (retrieved right now — use this to answer):\n"
                    f"  CPU: {state['cpu_percent']}%\n"
                    f"  RAM: {state['ram_used_gb']}/{state['ram_total_gb']} GB ({state['ram_percent']}%)\n"
                    f"  GPU: {state['gpu_name']} — VRAM {state['gpu_vram_used_mb']}/{state['gpu_vram_total_mb']} MB, Load {state['gpu_load_percent']}%\n"
                    f"{disk_lines}"
                    f"{battery_line}"
                    f"  Active window: {state['active_window']}\n\n"
                    f"User asked: {user_input}\n"
                    f"Answer using ONLY the live data above. Do NOT say you lack data."
                )
                _sys_prompt = "You are JARVIS, a helpful system monitor. Answer the user's question using the live system data provided. Be concise and direct."
                return await self.llm.generate(
                    prompt,
                    task_type=TaskType.GENERAL_KNOWLEDGE,
                    system_prompt=_sys_prompt
                )
            elif intent == "WEB_TASK":
                if self.web_agent:
                    return await self.web_agent.handle(user_input, context)
                return await self.llm.generate(
                    user_input,
                    task_type=TaskType.GENERAL_KNOWLEDGE,
                    system_prompt=system_prompt
                )
            elif intent == "DATA_TASK":
                if self.data_agent:
                    return await self.data_agent.handle(user_input, context)
                return await self.llm.generate(
                    user_input,
                    task_type=TaskType.CODE_WRITE,
                    system_prompt=system_prompt
                )
            elif intent == "EMAIL_TASK":
                if self.email_agent:
                    return await self.email_agent.handle(user_input, context)
                return "Email agent not available. Configure email in config.yaml."
            elif intent == "SCHEDULE_TASK":
                if self.scheduler_agent:
                    return await self.scheduler_agent.handle(user_input, context)
                return "Scheduler not available."
            elif intent == "API_TASK":
                if self.api_agent:
                    return await self.api_agent.handle(user_input, context)
                return await self.llm.generate(
                    user_input,
                    task_type=TaskType.GENERAL_KNOWLEDGE,
                    system_prompt=system_prompt
                )
            elif intent == "WORKFLOW":
                if self.workflow_agent:
                    return await self.workflow_agent.handle(user_input, context)
                # Fallback: treat as terminal task
                return await self.terminal_agent.execute(user_input, context)
            else:  # CHAT and fallback
                return await self.llm.generate(
                    user_input,
                    task_type=TaskType.SIMPLE_CHAT,
                    system_prompt=system_prompt
                )
        except Exception as exc:
            logger.error("Agent routing error for %s: %s", intent, exc)
            return f"Error during {intent}: {exc}"

    @staticmethod
    def _agent_to_intent(agent_name: str) -> str | None:
        """Map a brain-recommended agent name to the orchestrator intent label."""
        mapping = {
            "os_agent": "OS_COMMAND",
            "file_agent": "FILE_OP",
            "terminal_agent": "TERMINAL_TASK",
            "web_agent": "WEB_TASK",
            "email_agent": "EMAIL_TASK",
            "image_analyzer": "IMAGE_ANALYZE",
            "data_agent": "DATA_TASK",
            "scheduler_agent": "SCHEDULE_TASK",
            "api_agent": "API_TASK",
            "workflow_agent": "WORKFLOW",
        }
        return mapping.get(agent_name)

    @staticmethod
    def _is_brain_request(user_input: str) -> bool:
        """Check if the user is explicitly asking the brain to create/make a tool."""
        lower = user_input.lower()
        brain_triggers = [
            "create a tool", "make a tool", "build a tool",
            "create tool", "make tool", "build tool",
            "new tool", "generate a tool", "synthesize",
        ]
        return any(trigger in lower for trigger in brain_triggers)

    def _learn(
        self,
        user_input: str,
        response: str,
        intent: str,
        elapsed_ms: int,
    ) -> None:
        """Post-response learning step (runs in executor).

        Args:
            user_input: The user's text input.
            response: JARVIS response.
            intent: Classified intent.
            elapsed_ms: Response time in milliseconds.
        """
        try:
            self.interaction_logger.log(user_input, response, intent, elapsed_ms, cache_hit=False)
            self.memory.update(user_input, response)
            self.personal_model.update_from_interaction(user_input, response, intent)
        except Exception as exc:
            logger.error("Learning step failed: %s", exc)

    async def start_background_tasks(self) -> None:
        """Start background loops for context prefetch and memory flush."""
        asyncio.create_task(self._context_prefetch_loop())
        asyncio.create_task(self._memory_flush_loop())

    async def _context_prefetch_loop(self) -> None:
        """Pre-build context every 30 seconds based on the active app."""
        while True:
            try:
                await asyncio.sleep(30)
                active = self.os_agent.get_active_window()
                if active:
                    await self.context_builder.build_context(f"User is using {active}")
                self.cache.invalidate_system_state_cache()
            except Exception as exc:
                logger.debug("Context prefetch error: %s", exc)

    async def _memory_flush_loop(self) -> None:
        """Flush pending memory writes every 60 seconds."""
        while True:
            try:
                await asyncio.sleep(60)
                self.memory._save_to_disk()
            except Exception as exc:
                logger.debug("Memory flush error: %s", exc)

    # ── PARALLEL TASK RUNNER ──────────────────────────────────────────────────

    # Verbs that clearly start an independent subtask
    _TASK_STARTERS = {
        "search", "find", "look up", "google", "open", "launch", "start",
        "install", "run", "create", "build", "write", "read", "check",
        "kill", "close", "stop", "send", "message", "tell", "take",
        "show", "list", "get", "fetch", "download", "update",
    }

    def _split_parallel_tasks(self, text: str) -> list[str]:
        """Split 'do X and do Y' into independent subtask strings.

        Only splits when each part clearly begins with a known action verb,
        signalling they are independent operations.

        Args:
            text: User input text.

        Returns:
            List of subtask strings, or [text] if not clearly parallel.
        """
        # Split on " and " (not inside quotes)
        parts = re.split(r"\s+and\s+", text.strip(), maxsplit=2)
        if len(parts) < 2:
            return [text]

        valid: list[str] = []
        for part in parts:
            first = part.strip().split()[0].lower() if part.strip() else ""
            if any(part.strip().lower().startswith(v) for v in self._TASK_STARTERS):
                valid.append(part.strip())
            else:
                return [text]  # Part doesn't start with a task verb — not parallel

        return valid if len(valid) >= 2 else [text]

    async def _run_single_task(self, task: str) -> str:
        """Classify and execute a single subtask string.

        Args:
            task: Subtask text.

        Returns:
            Agent response string.
        """
        loop = asyncio.get_event_loop()
        intent_result = await loop.run_in_executor(
            self.executor, self.classifier.classify, task
        )
        intent, _ = intent_result
        context = await self.context_builder.build_context(task)
        return await self.route_to_agent(intent, task, context)

    async def _try_parallel_tasks(self, user_input: str) -> str | None:
        """Detect and run parallel independent subtasks.

        Returns combined result string, or None if input is not parallel.

        Args:
            user_input: Raw user input.

        Returns:
            Combined results string, or None if not a parallel task.
        """
        parts = self._split_parallel_tasks(user_input)
        if len(parts) < 2:
            return None

        # Classify each part to confirm they're genuinely different intents
        loop = asyncio.get_event_loop()
        intents = await asyncio.gather(*[
            loop.run_in_executor(self.executor, self.classifier.classify, p)
            for p in parts
        ])
        intent_labels = [i[0] for i in intents]

        # Only parallelize if at least two different intent types
        if len(set(intent_labels)) < 2:
            return None

        self.ui.console.print(
            f"\n  [bold magenta]>> Running {len(parts)} tasks in parallel...[/bold magenta]"
        )
        for i, (p, intent) in enumerate(zip(parts, intent_labels), 1):
            self.ui.console.print(f"  [dim]  Task {i}: [{intent}] {p}[/dim]")

        # Execute all tasks concurrently
        results = await asyncio.gather(*[
            self._run_single_task(p) for p in parts
        ], return_exceptions=True)

        combined: list[str] = []
        for i, (part, result) in enumerate(zip(parts, results), 1):
            if isinstance(result, Exception):
                combined.append(f"[Task {i}] {part}\n  Error: {result}")
            else:
                combined.append(f"[Task {i}] {part}\n  {result}")

        return "\n\n".join(combined)

    # ── SEQUENTIAL MULTI-STEP COMMAND RUNNER ──────────────────────────────

    _STEP_SEPARATORS = re.compile(
        r'\s+(?:then|after that|and then|afterwards|followed by)\s+',
        re.IGNORECASE,
    )

    # Dependency indicators — "X and Y [in it / through it / using it]"
    # means step 2 depends on the environment created by step 1.
    # Split on " and " when these trailing phrases are present.
    _DEPENDENCY_TRAIL = re.compile(
        r'\s+(?:in it|through it|using it|inside it|within it|via it|'
        r'through that|using that|in that|inside that|within that)\s*$',
        re.IGNORECASE,
    )

    # Phrases that indicate step 1 failed — stop sequential execution.
    _FAILURE_PHRASES = (
        "[llm timeout", "[llm error", "[llm fallback error",
        "error:", "failed", "could not", "unable to", "not found",
        "timed out", "exception", "traceback",
    )

    def _split_sequential_steps(self, text: str) -> list[str]:
        """Split 'do X then do Y' or 'do X and Y in it' into step strings.

        Handles both explicit connectors (then, after that) and implicit
        dependency indicators (in it, through it, using it).

        Args:
            text: User input text.

        Returns:
            List of step strings, or [text] if no split found.
        """
        # Explicit sequential connectors
        parts = self._STEP_SEPARATORS.split(text.strip())
        if len(parts) >= 2:
            return [p.strip() for p in parts if p.strip()]

        # Implicit dependency: "X and Y in it" → [X, Y]
        if self._DEPENDENCY_TRAIL.search(text):
            # Remove the trailing dependency phrase, then split on " and "
            clean = self._DEPENDENCY_TRAIL.sub("", text).strip()
            and_parts = re.split(r'\s+and\s+', clean, maxsplit=1, flags=re.I)
            if len(and_parts) == 2:
                return [p.strip() for p in and_parts if p.strip()]

        return [text]

    @staticmethod
    def _step_failed(result: str) -> bool:
        """Return True if a step result string indicates failure."""
        lower = result.lower()
        return any(phrase in lower for phrase in Orchestrator._FAILURE_PHRASES)

    async def _try_sequential_steps(self, user_input: str) -> str | None:
        """Detect and run sequential multi-step commands.

        Stops early if a step fails — dependent steps cannot run if their
        prerequisite failed (e.g., can't open browser inside a VPN that
        failed to start).

        Args:
            user_input: Raw user input.

        Returns:
            Combined results string, or None if not sequential steps.
        """
        steps = self._split_sequential_steps(user_input)
        if len(steps) < 2:
            return None

        is_dependent = bool(self._DEPENDENCY_TRAIL.search(user_input))
        self.ui.console.print(
            f"\n  [bold magenta]>> Running {len(steps)} steps sequentially"
            f"{' (dependent)' if is_dependent else ''}...[/bold magenta]"
        )

        results: list[str] = []
        for i, step in enumerate(steps, 1):
            self.ui.console.print(f"  [dim]  Step {i}: {step}[/dim]")
            try:
                result = await self._run_single_task(step)
                results.append(f"[Step {i}] {step}\n  {result}")
                # For dependent steps, stop if this step failed
                if is_dependent and self._step_failed(result):
                    results.append(
                        f"[Step {i+1}+] Skipped — Step {i} did not succeed. "
                        f"Cannot proceed with dependent steps."
                    )
                    break
            except Exception as exc:
                results.append(f"[Step {i}] {step}\n  Error: {exc}")
                if is_dependent:
                    results.append(
                        f"[Step {i+1}+] Skipped — Step {i} raised an exception."
                    )
                    break

        return "\n\n".join(results)
