"""LLMCore — Model router for JARVIS, supporting per-task model selection and logging."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_tokens: int
    timeout: int
    temperature: float
    layer: str  # "fast", "reasoning", "code", "vision"

class TaskType(Enum):
    # Layer 1 — Fast
    INTENT_CLASSIFY    = "intent_classify"
    SIMPLE_CHAT        = "simple_chat"
    FORMAT_RESULT      = "format_result"
    OS_CONFIRM         = "os_confirm"
    FILE_CONFIRM       = "file_confirm"
    SIMPLE_MATH        = "simple_math"
    TRANSLATION        = "translation"
    GENERAL_KNOWLEDGE  = "general_knowledge"
    SCHEDULE_CONFIRM   = "schedule_confirm"
    EMAIL_COMPOSE      = "email_compose"

    # Layer 2 — Reasoning
    TOOL_BLUEPRINT     = "tool_blueprint"
    TOOL_SELF_REVIEW   = "tool_self_review"
    REACT_PLANNING     = "react_planning"
    MULTI_STEP_PLAN    = "multi_step_plan"
    DECISION_ENGINE    = "decision_engine"
    COMPLEX_MATH       = "complex_math"
    COMPARISON         = "comparison"
    SESSION_LEARNING   = "session_learning"
    MEMORY_SYNTHESIS   = "memory_synthesis"

    # Layer 3 — Code
    TOOL_CODE_GEN      = "tool_code_gen"
    TOOL_BUG_FIX       = "tool_bug_fix"
    TOOL_IMPROVE       = "tool_improve"
    CODE_EXPLAIN       = "code_explain"
    CODE_WRITE         = "code_write"
    SHELL_CMD_GEN      = "shell_cmd_gen"
    REGEX_GEN          = "regex_gen"

    # Layer 4 — Vision
    SCREENSHOT_ANALYZE = "screenshot_analyze"
    IMAGE_DESCRIBE     = "image_describe"
    IMAGE_OCR          = "image_ocr"
    VISUAL_QA          = "visual_qa"
    CHART_ANALYZE      = "chart_analyze"



class LLMCore:
    """Model router for JARVIS. Selects the best model for each task type."""

    async def check_ollama_health(self) -> bool:
        """Check if the Ollama backend is running and reachable."""
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    _TASK_MODEL_MAP: dict[TaskType, str] = {
        # Layer 1 — all use fast model
        TaskType.INTENT_CLASSIFY   : "fast",
        TaskType.SIMPLE_CHAT       : "fast",
        TaskType.FORMAT_RESULT     : "fast",
        TaskType.OS_CONFIRM        : "fast",
        TaskType.FILE_CONFIRM      : "fast",
        TaskType.SIMPLE_MATH       : "fast",
        TaskType.TRANSLATION       : "fast",
        TaskType.GENERAL_KNOWLEDGE : "fast",
        TaskType.SCHEDULE_CONFIRM  : "fast",
        TaskType.EMAIL_COMPOSE     : "fast",

        # Layer 2 — all use reasoning model
        TaskType.TOOL_BLUEPRINT    : "reasoning",
        TaskType.TOOL_SELF_REVIEW  : "reasoning",
        TaskType.REACT_PLANNING    : "reasoning",
        TaskType.MULTI_STEP_PLAN   : "reasoning",
        TaskType.DECISION_ENGINE   : "reasoning",
        TaskType.COMPLEX_MATH      : "reasoning",
        TaskType.COMPARISON        : "reasoning",
        TaskType.SESSION_LEARNING  : "reasoning",
        TaskType.MEMORY_SYNTHESIS  : "reasoning",

        # Layer 3 — all use code model
        TaskType.TOOL_CODE_GEN     : "code",
        TaskType.TOOL_BUG_FIX      : "code",
        TaskType.TOOL_IMPROVE      : "code",
        TaskType.CODE_EXPLAIN      : "code",
        TaskType.CODE_WRITE        : "code",
        TaskType.SHELL_CMD_GEN     : "code",
        TaskType.REGEX_GEN         : "code",

        # Layer 4 — all use vision model
        TaskType.SCREENSHOT_ANALYZE: "vision",
        TaskType.IMAGE_DESCRIBE    : "vision",
        TaskType.IMAGE_OCR         : "vision",
        TaskType.VISUAL_QA         : "vision",
        TaskType.CHART_ANALYZE     : "vision",
    }

    # Per-task timeout overrides — take precedence over layer defaults.
    # Reasoning tasks that involve LLM chain-of-thought need more time.
    _TASK_TIMEOUT_OVERRIDES: dict[TaskType, int] = {
        TaskType.TOOL_BLUEPRINT   : 120,   # Full blueprint generation
        TaskType.TOOL_SELF_REVIEW : 90,    # Code review pass
        TaskType.TOOL_CODE_GEN    : 90,    # Code generation
        TaskType.DECISION_ENGINE  : 90,    # 6-step reasoning JSON
        TaskType.MULTI_STEP_PLAN  : 90,    # Workflow decomposition
        TaskType.SESSION_LEARNING : 60,    # Session analysis
        TaskType.MEMORY_SYNTHESIS : 60,    # Memory consolidation
    }

    @staticmethod
    def _extract_tool_result(prompt: str) -> str:
        """Pull the raw MCP/tool result text back out of a formatter prompt."""
        marker = "Tool result:\n"
        idx = prompt.find(marker)
        if idx == -1:
            return ""
        result = prompt[idx + len(marker):]
        result = result.split("\n\nSummarize this result naturally for the user.", 1)[0]
        return result.strip()

    def _timeout_fallback_response(
        self,
        task_type: TaskType,
        prompt: str,
        model_config: ModelConfig,
    ) -> str:
        """Return a user-safe fallback string instead of leaking raw timeout text."""
        if task_type == TaskType.INTENT_CLASSIFY:
            return "{}"
        if task_type == TaskType.FORMAT_RESULT:
            tool_result = self._extract_tool_result(prompt)
            return tool_result[:2000] if tool_result else "The tool finished, but formatting timed out."
        if task_type in (TaskType.SIMPLE_CHAT, TaskType.GENERAL_KNOWLEDGE, TaskType.MEMORY_SYNTHESIS):
            return "I'm having a short delay right now. Please try that again."
        if model_config.layer == "fast":
            return "That took too long to process. Please try again."
        return f"[LLM timeout: {model_config.name}]"

    def __init__(self, config: dict) -> None:
        """Initialize the model router from config.yaml."""
        self.config = config
        llm_cfg = config["llm"]
        self.base_url = llm_cfg["base_url"]
        self.keep_alive = llm_cfg.get("keep_alive", "5m")
        self.client = httpx.AsyncClient(timeout=120)

        self._models: dict[str, ModelConfig] = {
            "fast": ModelConfig(
                name=llm_cfg["model_fast"],
                max_tokens=llm_cfg["model_fast_max_tokens"],
                timeout=llm_cfg["model_fast_timeout"],
                temperature=llm_cfg["model_fast_temperature"],
                layer="fast",
            ),
            "reasoning": ModelConfig(
                name=llm_cfg["model_reasoning"],
                max_tokens=llm_cfg["model_reasoning_max_tokens"],
                timeout=llm_cfg["model_reasoning_timeout"],
                temperature=llm_cfg["model_reasoning_temperature"],
                layer="reasoning",
            ),
            "code": ModelConfig(
                name=llm_cfg["model_code"],
                max_tokens=llm_cfg["model_code_max_tokens"],
                timeout=llm_cfg["model_code_timeout"],
                temperature=llm_cfg["model_code_temperature"],
                layer="code",
            ),
            "vision": ModelConfig(
                name=llm_cfg["model_vision"],
                max_tokens=llm_cfg["model_vision_max_tokens"],
                timeout=llm_cfg["model_vision_timeout"],
                temperature=llm_cfg["model_vision_temperature"],
                layer="vision",
            ),
        }

    def _get_model_for_task(self, task_type: TaskType) -> ModelConfig:
        """Return the correct ModelConfig for this task type."""
        layer = self._TASK_MODEL_MAP.get(task_type, "fast")
        return self._models[layer]

    async def _get_installed_models(self) -> list[str]:
        """Return the list of available model names in Ollama."""
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return []

    async def verify_all_models(self) -> dict[str, bool]:
        """Check all 4 configured models exist in Ollama."""
        results = {}
        installed = await self._get_installed_models()
        for layer, config in self._models.items():
            results[layer] = config.name in installed
            if not results[layer]:
                logger.error(
                    "Model '%s' (layer: %s) not found in Ollama. Run: ollama pull %s",
                    config.name, layer, config.name
                )
        return results

    async def generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.SIMPLE_CHAT,
        system_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """Route to correct model based on task type and generate response."""
        model_config = self._get_model_for_task(task_type)
        payload = {
            "model": model_config.name,
            "messages": [],
            "stream": stream,
            "options": {
                "num_predict": model_config.max_tokens,
                "temperature": model_config.temperature,
                "top_p": 0.85,
                "top_k": 30,
                "repeat_penalty": 1.1,
            },
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        if image_path and model_config.layer == "vision":
            # Vision models expect image input as a special field
            payload["images"] = [image_path]
        payload["messages"].append({"role": "user", "content": prompt})

        effective_timeout = self._TASK_TIMEOUT_OVERRIDES.get(task_type, model_config.timeout)
        logger.info(
            "LLM call: task=%s layer=%s model=%s max_tokens=%d timeout=%ds",
            task_type.value, model_config.layer, model_config.name, model_config.max_tokens, effective_timeout
        )

        start = time.time()
        try:
            resp = await asyncio.wait_for(
                self.client.post(f"{self.base_url}/api/chat", json=payload),
                timeout=effective_timeout
            )
            if resp.status_code != 200:
                logger.error("Ollama 500 detail: %s", resp.text[:500])
            resp.raise_for_status()
            data = resp.json()
            result = data.get("message", {}).get("content", "")
            elapsed = time.time() - start
            logger.info(
                "LLM: task=%s layer=%s model=%s tokens=%d time=%.1fs",
                task_type.value, model_config.layer, model_config.name, model_config.max_tokens, elapsed
            )
            return result
        except asyncio.TimeoutError:
            if model_config.layer == "reasoning":
                logger.warning(
                    "Reasoning model timed out, falling back to fast model for task %s.",
                    task_type.value
                )
                # Fallback to fast model
                fast_config = self._models["fast"]
                payload["model"] = fast_config.name
                payload["options"]["num_predict"] = fast_config.max_tokens
                payload["options"]["temperature"] = fast_config.temperature
                try:
                    resp = await asyncio.wait_for(
                        self.client.post(f"{self.base_url}/api/chat", json=payload),
                        timeout=60  # fallback needs time to load from disk
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    result = data.get("message", {}).get("content", "")
                    elapsed = time.time() - start
                    logger.info(
                        "LLM: task=%s layer=fast model=%s tokens=%d time=%.1fs (fallback)",
                        task_type.value, fast_config.name, fast_config.max_tokens, elapsed
                    )
                    return result
                except Exception as exc:
                    logger.error("LLM fallback error: %s", exc)
                    return f"[LLM fallback error: {exc}]"
            logger.error("LLM timeout for model %s (task %s, timeout=%ds)", model_config.name, task_type.value, effective_timeout)
            return self._timeout_fallback_response(task_type, prompt, model_config)
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            return f"[LLM error: {exc}]"

    async def chat(
        self,
        messages: list[dict],
        task_type: TaskType = TaskType.SHELL_CMD_GEN,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **_kwargs,
    ) -> str:
        """Send a raw messages list to the model for this task type.

        Used by agents that manage their own conversation history (e.g. terminal_agent).

        Args:
            messages: Full messages list with role/content dicts.
            task_type: Determines which model layer to use.
            max_tokens: Override max tokens if provided.
            temperature: Override temperature if provided.

        Returns:
            Model response string.
        """
        model_config = self._get_model_for_task(task_type)
        payload = {
            "model": model_config.name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens or model_config.max_tokens,
                "temperature": temperature if temperature is not None else model_config.temperature,
                "top_p": 0.85,
                "top_k": 30,
                "repeat_penalty": 1.1,
            },
        }
        chat_timeout = self._TASK_TIMEOUT_OVERRIDES.get(task_type, model_config.timeout)
        logger.info(
            "LLM chat: task=%s layer=%s model=%s timeout=%ds",
            task_type.value, model_config.layer, model_config.name, chat_timeout,
        )
        start = time.time()
        try:
            resp = await asyncio.wait_for(
                self.client.post(f"{self.base_url}/api/chat", json=payload),
                timeout=chat_timeout,
            )
            if resp.status_code != 200:
                logger.error("Ollama error: %s", resp.text[:300])
            resp.raise_for_status()
            elapsed = time.time() - start
            result = resp.json().get("message", {}).get("content", "")
            logger.info(
                "LLM chat: task=%s model=%s time=%.1fs",
                task_type.value, model_config.name, elapsed,
            )
            return result
        except asyncio.TimeoutError:
            logger.error("LLM chat timeout: %s", model_config.name)
            return ""
        except Exception as exc:
            logger.error("LLM chat error: %s", exc)
            return ""

    async def display_model_status(self) -> None:
        """Print a startup table showing model availability for all 4 layers."""
        results = await self.verify_all_models()
        width = 55
        border = "─" * width
        print(f"┌{border}┐")
        print(f"│{'🤖 Model Configuration':^{width}}│")
        print(f"│{' ' * width}│")
        rows = [
            ("Layer 1 Fast",      "fast"),
            ("Layer 2 Reasoning", "reasoning"),
            ("Layer 3 Code",      "code"),
            ("Layer 4 Vision",    "vision"),
        ]
        for label, layer in rows:
            cfg = self._models[layer]
            ok = results.get(layer, False)
            status = "✅ Ready  " if ok else "❌ Missing"
            line = f"  {label:<18} {cfg.name:<22} {status}"
            print(f"│{line:<{width}}│")
            if not ok:
                fix = f"  Fix: ollama pull {cfg.name}"
                print(f"│{fix:<{width}}│")
        print(f"│{' ' * width}│")
        keep = f"  Keep alive: {self.keep_alive} after last use"
        print(f"│{keep:<{width}}│")
        print(f"└{border}┘")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
