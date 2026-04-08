"""Context Builder — Assembles full prompts with memory, state, and personal model."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Injected into every system prompt to prevent hallucinated live data ────────
_REALTIME_RULES: str = """
REAL-TIME DATA RULES — NON-NEGOTIABLE:
You are a REASONING and PLANNING engine only.
You have ZERO access to live data of any kind.
Your training data is static and outdated.

STRICT RULES:
1. NEVER state a temperature, price, exchange rate, score,
   or any real-time value from your own knowledge
2. NEVER claim you fetched from OpenWeatherMap, checked
   the internet, or any similar statement — you cannot
   make HTTP calls
3. If api_agent returned data in THIS conversation →
   present it and always say:
   "📡 Source: [source name] | Fetched at [time]"
4. Rule 4 applies ONLY when the user explicitly asked for
   weather, current prices, exchange rates, live news, or
   live scores AND no api_agent data was returned:
   → say "I need live data for that. Routing to real-time agent."
   Do NOT say this for system tasks, tool failures, or app launches.
5. If a tool creation or system task failed → say:
   "I was unable to complete that task. Please try a simpler
   command or rephrase your request."
6. Violating rules 1 or 2 is a CRITICAL FAILURE
""".strip()


class ContextBuilder:
    """Assembles the full system prompt for every LLM call.

    Injects who the user is, what they are doing, and what they care about.
    """

    def __init__(
        self,
        memory_system,
        personal_model,
        interaction_logger,
        os_agent,
        config: dict,
        tool_registry=None,
    ) -> None:
        """Initialize the context builder.

        Args:
            memory_system: MemorySystem instance.
            personal_model: PersonalModel instance.
            interaction_logger: InteractionLogger instance.
            os_agent: OSAgent instance.
            config: Full JARVIS configuration dictionary.
            tool_registry: Optional brain ToolRegistry for tool context injection.
        """
        self.memory = memory_system
        self.personal_model = personal_model
        self.logger = interaction_logger
        self.os_agent = os_agent
        self.max_context = config["memory"]["max_context_interactions"]
        self.tool_registry = tool_registry

        # Load session lessons from personal model (once at init)
        self._learned_lessons: list[str] = self._load_lessons(config)

    # System prompts per task mode
    _SYSTEM_PROMPTS: dict[str, str] = {
        "chat": (
            "You are JARVIS, a smart AI assistant on Windows 11. "
            "Faheem is your boss — call him Faheem or Boss. "
            "Be direct and concise. Answer only what was asked.\n\n"
            "WHEN TO ANSWER DIRECTLY (no tools needed):\n"
            "- Explanations: 'what is machine learning', 'explain recursion'\n"
            "- Definitions: 'define polymorphism', 'meaning of API'\n"
            "- How-to knowledge: 'how does a CPU work', 'how to learn Python'\n"
            "- Comparisons: 'difference between Java and Python'\n"
            "- General knowledge from your training data\n\n"
            "WHEN TO USE TOOLS (need live data):\n"
            "- Weather, prices, scores, exchange rates, current news\n"
            "- Anything that changes in real time"
        ),
        "code": (
            "You are JARVIS, a senior software engineer assistant on Windows 11.\n"
            "Faheem is your boss.\n\n"
            "When handling code tasks:\n"
            "- Provide complete, working code — never truncate\n"
            "- Use markdown code blocks with the language (```python, ```js, etc.)\n"
            "- Explain what you changed and why, briefly\n"
            "- Point out bugs or issues you notice\n"
            "- Use proper error handling\n"
            "- If you need to read a file to answer, say: READ: <filename>"
        ),
        "system": (
            "You are JARVIS, a system monitoring assistant on Windows 11. "
            "Faheem is your boss. Be concise. Report system info clearly with numbers."
        ),
        "web": (
            "You are JARVIS, an AI assistant. Faheem is your boss. "
            "Summarize web information clearly and concisely."
        ),
    }

    async def build_context(self, user_input: str, mode: str = "chat") -> dict:
        """Build the full context dict for a request.

        Args:
            user_input: The user's input text.
            mode: LLM mode (chat, code, system, web).

        Returns:
            Dict with system_prompt, system_state, relevant_memory,
            personal_context, and optionally brain_tools.
        """
        system_state = self.os_agent.get_system_state()
        relevant_memory = self.memory.query(user_input, top_k=5)
        personal_context = self.personal_model.get_context_string()
        recent = self.logger.get_recent(self.max_context)

        system_prompt = self._SYSTEM_PROMPTS.get(mode, self._SYSTEM_PROMPTS["chat"])

        # Inject current date/time so the LLM can answer time-related questions
        now = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        system_prompt = f"Current date and time: {now}\nYou MUST use this date/time when asked about today's date or current time.\n\n{system_prompt}"

        # Inject real-time data rules to prevent hallucinated live data
        system_prompt += f"\n\n{_REALTIME_RULES}"

        # Inject learned lessons (max 5, high confidence only)
        if self._learned_lessons:
            lessons_block = "\n".join(
                f"- {lesson}" for lesson in self._learned_lessons
            )
            system_prompt += f"\n\nLearned from past sessions:\n{lessons_block}"

        # Brain tools are handled by the DecisionEngine, not the LLM.
        # Injecting them into the system prompt confuses small models
        # into hallucinating tool execution output for simple queries.

        return {
            "system_prompt": system_prompt,
            "system_state": system_state,
            "relevant_memory": relevant_memory,
            "personal_context": personal_context,
        }

    def assemble_system_prompt(
        self,
        system_state: dict,
        relevant_memory: list[dict],
        personal_context: str,
        current_datetime: str,
        recent_interactions: list[dict] | None = None,
    ) -> str:
        """Build the full system prompt string.

        Args:
            system_state: Current system metrics dict.
            relevant_memory: Top relevant past interactions.
            personal_context: Formatted personal model string.
            current_datetime: Current date/time string.
            recent_interactions: Recent interaction log entries.

        Returns:
            Complete system prompt string.
        """
        # Format memory
        memory_lines = ""
        if relevant_memory:
            for i, mem in enumerate(relevant_memory[:5], 1):
                memory_lines += f"  {i}. [{mem.get('category', '')}] {mem.get('text', '')[:200]}\n"
        else:
            memory_lines = "  No relevant memories found.\n"

        # Format recent interactions
        recent_lines = ""
        if recent_interactions:
            for interaction in recent_interactions[:5]:
                recent_lines += (
                    f"  - User: {interaction.get('user_input', '')[:100]}\n"
                    f"    JARVIS: {interaction.get('response', '')[:100]}\n"
                )

        prompt = (
            "You are JARVIS, an AI assistant on Windows 11. "
            "Faheem is your boss — always call him Faheem or Boss. "
            "Be concise (1-3 sentences). Answer only what was asked. "
            "Never invent tasks."
            f"\n\n{_REALTIME_RULES}"
        )

        return prompt

    @staticmethod
    def _load_lessons(config: dict) -> list[str]:
        """Load high-confidence lessons from personal model at startup.

        Args:
            config: Full JARVIS configuration dictionary.

        Returns:
            List of lesson strings (max 5).
        """
        try:
            from brain.session_learner import SessionLearner

            personal_model_path = Path(
                config.get("memory", {}).get(
                    "personal_model_path", "data/personal_model.json"
                )
            )
            brain_db_path = Path(
                config.get("brain", {}).get("db_path", "data/brain.db")
            )
            lessons = SessionLearner.load_lessons_for_context(
                personal_model_path, brain_db_path
            )
            if lessons:
                logger.info(
                    "ContextBuilder: loaded %d lessons from past sessions",
                    len(lessons),
                )
            return lessons
        except Exception as exc:
            logger.debug("Could not load session lessons: %s", exc)
            return []
