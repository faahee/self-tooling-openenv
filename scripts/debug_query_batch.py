"""Run a safe 12-query routing/debug batch against the real orchestrator stack."""
from __future__ import annotations

import asyncio
import copy
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.file_agent import FileAgent
from agents.os_agent import OSAgent
from agents.workflow_agent import WorkflowAgent
from core.intent_classifier import IntentClassifier
from core.mcp_client import MCPClient
from core.orchestrator import Orchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


class DummyConsole:
    def print(self, *args, **kwargs) -> None:
        return None


class DummyUI:
    def __init__(self) -> None:
        self.console = DummyConsole()

    def display_user_input(self, text: str, source: str = "text") -> None:
        return None

    def display_response(self, response: str, intent: str | None = None) -> None:
        return None

    def speak(self, text: str) -> None:
        return None

    def notify(self, title: str, message: str) -> None:
        return None

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None
        return _noop


class DummyCache:
    def get(self, text: str):
        return None

    def set(self, text: str, response: str) -> None:
        return None


class DummyLogger:
    def log(self, *args, **kwargs) -> None:
        return None


class DummyMemory:
    def query(self, text: str, top_k: int = 5):
        return []

    def add_memory(self, text: str, category: str = "fact") -> None:
        return None

    def update(self, *args, **kwargs) -> None:
        return None


class DummyContextBuilder:
    async def build_context(self, user_input: str, mode: str = "chat") -> dict:
        return {"system_prompt": "", "mode": mode}


class DummyPersonalModel:
    def update_from_interaction(self, *args, **kwargs) -> None:
        return None


class DummyLLM:
    async def generate(self, prompt: str, task_type=None, system_prompt: str = "") -> str:
        low = prompt.lower()
        if "return only a json object" in low or "return json only" in (system_prompt or "").lower():
            return "{}"
        if "summarize this result naturally" in low:
            marker = "tool result:"
            idx = low.find(marker)
            if idx != -1:
                return prompt[idx + len(marker):].strip()[:300]
            return "Tool executed."
        if "multi_step_plan" in str(task_type):
            return "Step 1: do the task."
        return "I was unable to complete that task. Please try a simpler command or rephrase your request."

    async def close(self) -> None:
        return None


class DummyTerminalAgent:
    async def execute(self, user_input: str, context: dict) -> str:
        return f"[terminal_fallback] {user_input}"


class DummyImageAnalyzer:
    last_screenshot = None

    def build_prompt(self, user_input: str) -> str:
        return user_input

    async def analyze_screenshot(self, prompt: str) -> str:
        return f"[image_analyze] {prompt}"

    async def analyze_file(self, path: str, prompt: str) -> str:
        return f"[image_analyze_file] {path}: {prompt}"


TEST_CASES = [
    {
        "query": "minimize all screens",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "show me the desktop",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "open browser",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "turn up volume",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "pause",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "next song",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "take screenshot",
        "intent": "OS_COMMAND",
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "where is java file",
        "intent": "FILE_OP",
        "must_not": ["I need live data for that", "[terminal_fallback]"],
    },
    {
        "query": "where is my document",
        "intent": "FILE_OP",
        "must_not": ["I need live data for that", "[terminal_fallback]"],
    },
    {
        "query": "create a folder name kool in desktop",
        "intent": "FILE_OP",
        "must_not": ["Workflow complete", "I need live data for that"],
        "must_contain": ["Created folder", "desktop", "kool"],
    },
    {
        "query": "make a new folder in downloads",
        "intent": "FILE_OP",
        "must_not": ["Workflow complete", "I need live data for that", "Could not determine filename"],
        "must_contain": ["Created folder", "downloads", "new folder"],
    },
    {
        "query": "create a folder name kool and put it in desktop",
        "intent": "FILE_OP",
        "must_not": ["Workflow complete", "I need live data for that"],
        "must_contain": ["Created folder", "desktop", "kool"],
    },
]


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("ui", {})
    cfg["ui"]["tts_enabled"] = False
    cfg["ui"]["stream_display"] = False
    cfg.setdefault("mcp", {})
    cfg["mcp"]["enabled"] = True
    servers = cfg["mcp"].get("servers", {})
    for name, server in servers.items():
        server["enabled"] = name == "windows_mcp"
    return cfg


def patch_side_effects(os_agent: OSAgent, file_agent: FileAgent) -> None:
    import agents.os_agent as os_mod

    class _FakeImage:
        def save(self, path: str) -> None:
            Path(path).write_bytes(b"fake-png")

    os_mod.pyautogui.hotkey = lambda *args, **kwargs: None
    os_mod.pyautogui.press = lambda *args, **kwargs: None
    os_mod.pyautogui.typewrite = lambda *args, **kwargs: None
    os_mod.pyautogui.screenshot = lambda *args, **kwargs: _FakeImage()
    os_agent.start_process = lambda app_name: True  # type: ignore[method-assign]
    os_agent.set_volume = lambda level: True  # type: ignore[method-assign]
    file_agent.create_directory = lambda path: True  # type: ignore[method-assign]


async def build_orchestrator() -> tuple[Orchestrator, IntentClassifier, MCPClient]:
    config = load_config()
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    classifier = IntentClassifier(config, embedder=embedder)
    os_agent = OSAgent(config)
    file_agent = FileAgent(config)
    patch_side_effects(os_agent, file_agent)

    ui = DummyUI()
    llm = DummyLLM()
    orchestrator = Orchestrator(
        config=config,
        intent_classifier=classifier,
        context_builder=DummyContextBuilder(),
        llm_core=llm,
        os_agent=os_agent,
        file_agent=file_agent,
        image_analyzer=DummyImageAnalyzer(),
        terminal_agent=DummyTerminalAgent(),
        memory_system=DummyMemory(),
        personal_model=DummyPersonalModel(),
        interaction_logger=DummyLogger(),
        response_cache=DummyCache(),
        ui_layer=ui,
        web_agent=None,
        checkpoint_manager=None,
        data_agent=None,
        email_agent=None,
        api_agent=None,
        workflow_agent=None,
    )
    workflow_agent = WorkflowAgent(config, orchestrator.handle_input, llm, ui, web_agent=None)
    orchestrator.workflow_agent = workflow_agent

    mcp_client = MCPClient(config)
    if mcp_client.enabled:
        await mcp_client.connect_all()
    orchestrator.mcp_client = mcp_client
    return orchestrator, classifier, mcp_client


def check_case(
    expected_intent: str,
    classified_intent: str,
    response: str,
    must_not: list[str],
    must_contain: list[str] | None = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if classified_intent != expected_intent:
        issues.append(f"classified as {classified_intent}, expected {expected_intent}")
    for marker in must_not:
        if marker.lower() in response.lower():
            issues.append(f"response contains forbidden marker: {marker}")
    for marker in must_contain or []:
        if marker.lower() not in response.lower():
            issues.append(f"response missing expected marker: {marker}")
    return (not issues), issues


async def main() -> None:
    orchestrator, classifier, mcp_client = await build_orchestrator()
    passes = 0
    failures: list[dict] = []

    try:
        print("=== Query Batch Debug ===")
        print(f"windows_mcp_connected={any(s.connected for s in mcp_client.servers.values())}")
        print(f"windows_mcp_tools={len(mcp_client.get_all_tools())}")
        print()

        for idx, case in enumerate(TEST_CASES, 1):
            query = case["query"]
            classified_intent, confidence = classifier.classify(query)
            workflow_flag = WorkflowAgent.is_workflow(query)
            response = await orchestrator.handle_input(query)
            ok, issues = check_case(
                case["intent"],
                classified_intent,
                response,
                case["must_not"],
                case.get("must_contain"),
            )

            if ok:
                passes += 1
            else:
                failures.append(
                    {
                        "query": query,
                        "classified_intent": classified_intent,
                        "confidence": round(confidence, 2),
                        "workflow": workflow_flag,
                        "response": response,
                        "issues": issues,
                    }
                )

            print(f"[{idx:02d}] {query}")
            print(f"  classify={classified_intent} ({confidence:.2f}) workflow={workflow_flag}")
            print(f"  response={response[:240]}")
            print(f"  verdict={'PASS' if ok else 'FAIL'}")
            if issues:
                print(f"  issues={' | '.join(issues)}")
            print()

        print(f"SUMMARY: {passes}/{len(TEST_CASES)} passed")
        if failures:
            print("FAILURES:")
            for failure in failures:
                print(f"- {failure['query']}: {' | '.join(failure['issues'])}")
    finally:
        if mcp_client.enabled:
            await mcp_client.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
