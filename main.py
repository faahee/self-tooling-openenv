"""JARVIS — Personal AI OS Agent entry point.

Initializes all modules and starts the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows so emojis don't crash the terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import uvicorn
import yaml
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis")


def load_config() -> dict:
    """Load .env and config.yaml.

    Returns:
        Parsed config dict.
    """
    load_dotenv()
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found. Run setup.py first.")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_data_dir() -> None:
    """Create the data/ directory tree if it doesn't exist."""
    for d in ["data", "data/screenshots"]:
        Path(d).mkdir(parents=True, exist_ok=True)


async def main() -> None:
    """Initialize all modules and start JARVIS."""
    config = load_config()
    ensure_data_dir()

    # ── 0. Shared CPU Embedder (VRAM fully reserved for Ollama LLM) ──────────
    from sentence_transformers import SentenceTransformer
    shared_embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    logger.info("Shared embedder loaded on cpu (VRAM reserved for LLM).")

    # ── 1. Interaction Logger (database first) ────────────────────────────────
    from memory.interaction_logger import InteractionLogger
    interaction_logger = InteractionLogger(config)
    logger.info("InteractionLogger ready.")

    # ── 2. Memory System ──────────────────────────────────────────────────────
    from memory.memory_system import MemorySystem
    memory_system = MemorySystem(config, embedder=shared_embedder)
    logger.info("MemorySystem ready.")

    # ── 3. Personal Model ─────────────────────────────────────────────────────
    from memory.personal_model import PersonalModel
    personal_model = PersonalModel(config)
    logger.info("PersonalModel ready.")

    # ── 4. Response Cache ─────────────────────────────────────────────────────
    from core.response_cache import ResponseCache
    response_cache = ResponseCache(config, embedder=shared_embedder)
    logger.info("ResponseCache ready.")

    # ── 5. Intent Classifier ──────────────────────────────────────────────────
    from core.intent_classifier import IntentClassifier
    intent_classifier = IntentClassifier(config, embedder=shared_embedder)
    logger.info("IntentClassifier ready.")

    # ── 6. OS Agent ───────────────────────────────────────────────────────────
    from agents.os_agent import OSAgent
    os_agent = OSAgent(config)
    logger.info("OSAgent ready.")

    # ── 7. File Agent ─────────────────────────────────────────────────────────
    from agents.file_agent import FileAgent
    file_agent = FileAgent(config)
    file_agent.start_watching()
    logger.info("FileAgent ready (watching directories).")

    # ── 8. Image Analyzer ─────────────────────────────────────────────────────
    from agents.image_analyzer import ImageAnalyzer
    image_analyzer = ImageAnalyzer(config)
    logger.info("ImageAnalyzer ready.")

    # Wire vision into OS agent so it can use moondream for GUI automation
    os_agent.vision = image_analyzer

    # ── 9. Context Builder ────────────────────────────────────────────────────
    from core.context_builder import ContextBuilder
    context_builder = ContextBuilder(
        memory_system, personal_model, interaction_logger, os_agent, config
    )
    logger.info("ContextBuilder ready.")

    # ── 9b. Brain — Self-evolving tool system ─────────────────────────────────
    decision_engine = None
    try:
        from brain.tool_registry import ToolRegistry
        from brain.code_validator import CodeValidator
        from brain.sandbox_executor import SandboxExecutor
        from brain.tool_synthesizer import ToolSynthesizer
        from brain.tool_loader import ToolLoader
        from brain.decision_engine import DecisionEngine

        tool_registry = ToolRegistry(config, embedder=shared_embedder)
        code_validator = CodeValidator(config)
        sandbox_executor = SandboxExecutor(config)
        logger.info("Brain core components initialized.")
    except Exception as brain_exc:
        logger.warning("Brain initialization deferred (LLM needed): %s", brain_exc)
        tool_registry = None

    # Inject tool registry into context builder for tool-aware prompts
    if tool_registry:
        context_builder.tool_registry = tool_registry

    # ── 10. LLM Core ─────────────────────────────────────────────────────────
    from core.llm_core import LLMCore
    llm_core = LLMCore(config)

    # Health check
    healthy = await llm_core.check_ollama_health()
    if not healthy:
        logger.error(
            "Ollama is not running! Start it with: ollama serve"
        )
        sys.exit(1)

    # Verify all configured models exist
    model_status = await llm_core.verify_all_models()
    missing = [layer for layer, ok in model_status.items() if not ok]
    if missing:
        logger.error(
            "Missing required model(s) for layer(s): %s.\nCheck config.yaml and run: ollama pull <model> for each missing model.",
            ", ".join(missing)
        )
        sys.exit(1)
    logger.info("LLMCore ready. All models available: %s", ", ".join(model_status.keys()))

    # ── 10b. MCP Client (Model Context Protocol) ─────────────────────────────
    from core.mcp_client import MCPClient
    mcp_client = MCPClient(config)
    if mcp_client.enabled:
        mcp_results = await mcp_client.connect_all()
        connected = sum(1 for v in mcp_results.values() if v)
        total_tools = sum(len(s.tools) for s in mcp_client.servers.values() if s.connected)
        logger.info(
            "MCPClient ready (%d servers, %d tools).", connected, total_tools
        )
    else:
        logger.info("MCPClient disabled in config.")

    # ── 11. UI Layer ──────────────────────────────────────────────────────────
    from ui.ui_layer import UILayer
    ui_layer = UILayer(config)
    ui_layer.show_banner()
    logger.info("UILayer ready.")

    # ── 11b. Brain — Finish initialization (needs LLM + UI) ──────────────────
    if tool_registry:
        try:
            tool_synthesizer = ToolSynthesizer(
                config, llm_core, code_validator, sandbox_executor, tool_registry, ui_layer
            )
            tool_loader = ToolLoader(config)
            decision_engine = DecisionEngine(
                config, tool_registry, tool_synthesizer, tool_loader,
                llm_core, ui_layer, interaction_logger, shared_embedder,
            )
            logger.info("Brain Decision Engine ready (self-evolving tools active).")
        except Exception as brain_exc:
            logger.warning("Brain Decision Engine failed to initialize: %s", brain_exc)
            decision_engine = None

    # ── 11c. Memory Auditor (needs tool_registry + interaction_logger + UI) ──
    from brain.memory_auditor import MemoryAuditor
    memory_auditor = MemoryAuditor(
        config,
        tool_registry=tool_registry,
        interaction_logger=interaction_logger,
        ui_layer=ui_layer,
    )
    await memory_auditor.start()
    logger.info("MemoryAuditor ready (startup scan done, background task active).")

    # ── 11d. Session Learner (needs interaction_logger + personal_model + UI)
    from brain.session_learner import SessionLearner
    session_learner = SessionLearner(
        config, interaction_logger, personal_model, ui_layer
    )
    logger.info("SessionLearner ready (will run on shutdown).")

    # ── 12. Checkpoint Manager ────────────────────────────────────────────────
    from core.checkpoint import CheckpointManager
    checkpoint_manager = CheckpointManager()
    logger.info("CheckpointManager ready (undo support active).")

    # ── 12b. Terminal Agent ───────────────────────────────────────────────────
    from agents.terminal_agent import TerminalAgent
    terminal_agent = TerminalAgent(config, llm_core, ui_layer, checkpoint_manager)
    logger.info("TerminalAgent ready (Claude Code mode + undo support).")

    # ── 12c. Web Agent ────────────────────────────────────────────────────────
    from agents.web_agent import WebAgent
    web_agent = WebAgent(config)
    logger.info("WebAgent ready (DuckDuckGo search + web fetch).")

    # ── 12d. Data Agent ───────────────────────────────────────────────────────
    from agents.data_agent import DataAgent
    data_agent = DataAgent(config, llm_core, ui_layer)
    logger.info("DataAgent ready (CSV/Excel/JSON analysis + charts).")

    # ── 12e. Email Agent ──────────────────────────────────────────────────────
    from agents.email_agent import EmailAgent
    email_agent = EmailAgent(config, llm_core, ui_layer)
    logger.info("EmailAgent ready (SMTP/IMAP — configure email in config.yaml).")

    # ── 12f. API Agent ────────────────────────────────────────────────────────
    from agents.api_agent import APIAgent
    api_agent = APIAgent(config, llm_core, ui_layer)
    logger.info("APIAgent ready (weather, crypto, currency, news, wikipedia, IP, joke, GitHub).")

    # ── 13. Voice Listener ────────────────────────────────────────────────────
    from perception.voice_listener import VoiceListener

    # The orchestrator is created next — we need a reference for the callback
    orchestrator_ref: list = []  # mutable container for late binding

    def on_voice_input(text: str) -> None:
        """Voice transcription callback — sends text to orchestrator."""
        if orchestrator_ref:
            orch = orchestrator_ref[0]
            asyncio.run_coroutine_threadsafe(
                orch.handle_input(text), loop
            )

    voice_listener = VoiceListener(config, on_transcription_callback=on_voice_input)
    try:
        voice_listener.start()
        logger.info("VoiceListener ready (wake word active).")
    except Exception as exc:
        logger.warning("VoiceListener could not start (no microphone?): %s", exc)

    # ── 14. Orchestrator ──────────────────────────────────────────────────────
    from core.orchestrator import Orchestrator, app
    orchestrator = Orchestrator(
        config=config,
        intent_classifier=intent_classifier,
        context_builder=context_builder,
        llm_core=llm_core,
        os_agent=os_agent,
        file_agent=file_agent,
        image_analyzer=image_analyzer,
        terminal_agent=terminal_agent,
        memory_system=memory_system,
        personal_model=personal_model,
        interaction_logger=interaction_logger,
        response_cache=response_cache,
        ui_layer=ui_layer,
        web_agent=web_agent,
        checkpoint_manager=checkpoint_manager,
        data_agent=data_agent,
        email_agent=email_agent,
        api_agent=api_agent,
    )
    # Attach brain Decision Engine if available
    if decision_engine:
        orchestrator.decision_engine = decision_engine

    # ── 14c. ReAct Engine (needs orchestrator + all brain components) ─────────
    react_cfg = config.get("react", {})
    if react_cfg.get("enabled", False) and decision_engine and tool_registry:
        try:
            from core.react_engine import ReActEngine
            react_engine = ReActEngine(
                config=config,
                llm_core=llm_core,
                tool_registry=tool_registry,
                tool_synthesizer=tool_synthesizer,
                tool_loader=tool_loader,
                decision_engine=decision_engine,
                code_validator=code_validator,
                sandbox_executor=sandbox_executor,
                ui_layer=ui_layer,
                orchestrator=orchestrator,
                interaction_logger=interaction_logger,
            )
            orchestrator.react_engine = react_engine
            logger.info("ReActEngine ready (Reason-Act-Observe loop active).")
        except Exception as react_exc:
            logger.warning("ReActEngine failed to initialize: %s", react_exc)

    # Attach MCP Client for tool routing
    orchestrator.mcp_client = mcp_client
    if mcp_client.enabled and tool_registry:
        tool_synthesizer.mcp_client = mcp_client
    # Attach Memory Auditor for on-demand "clean your memory" commands
    orchestrator.memory_auditor = memory_auditor
    # Attach Session Learner for on-demand "what did you learn" commands
    orchestrator.session_learner = session_learner
    orchestrator_ref.append(orchestrator)

    # ── 14b. Scheduler + Workflow (need orchestrator.handle_input ref) ────────
    from agents.scheduler_agent import SchedulerAgent
    scheduler_agent = SchedulerAgent(config, orchestrator.handle_input, ui_layer)
    scheduler_agent.start()
    orchestrator.scheduler_agent = scheduler_agent
    logger.info("SchedulerAgent ready (APScheduler running).")

    from agents.workflow_agent import WorkflowAgent
    workflow_agent = WorkflowAgent(config, orchestrator.handle_input, llm_core, ui_layer, web_agent=web_agent)
    orchestrator.workflow_agent = workflow_agent
    logger.info("WorkflowAgent ready (multi-step goal decomposition).")

    # Start background tasks
    await orchestrator.start_background_tasks()

    # ── Startup announcements ─────────────────────────────────────────────────
    sys_state = os_agent.get_system_state()
    gpu_name = sys_state.get("gpu_name", "Unknown")
    vram = sys_state.get("gpu_vram_total_mb", 0)
    ui_layer.show_status(
        f"CPU: {sys_state['cpu_percent']}% | "
        f"RAM: {sys_state['ram_used_gb']}/{sys_state['ram_total_gb']} GB | "
        f"GPU: {gpu_name} ({vram} MB VRAM) | "
        f"Disk free: {sys_state['disk_free_gb']} GB"
    )

    startup_msg = (
        f"JARVIS online. All systems operational. "
        f"{gpu_name} detected. {vram} megabytes of VRAM available. Ready."
    )
    ui_layer.speak(startup_msg)
    ui_layer.console.print(f"\n[bold bright_green]{startup_msg}[/bold bright_green]\n")

    # ── Terminal input loop (runs alongside FastAPI) ──────────────────────────
    loop = asyncio.get_event_loop()

    async def terminal_input_loop() -> None:
        """Accept text commands from the terminal."""
        while True:
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: input("JARVIS > ")
                )
                text = user_input.strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit", "bye"):
                    ui_layer.speak("Goodbye.")
                    ui_layer.console.print("[bold bright_blue]JARVIS shutting down.[/bold bright_blue]")
                    break
                # Quick undo shortcut — also handled inside orchestrator
                if text.lower() in ("undo", "u"):
                    result = checkpoint_manager.undo()
                    ui_layer.console.print(f"  [bold green]{result}[/bold green]")
                    continue
                await orchestrator.handle_input(text)
            except (EOFError, KeyboardInterrupt):
                break

    # Run FastAPI server and terminal input concurrently
    port = config["orchestrator"]["port"]
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            logger.warning("Port %d in use, trying %d", port, port + 1)
            port += 1
    server_config = uvicorn.Config(
        app,
        host=config["orchestrator"]["host"],
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            server.serve(),
            terminal_input_loop(),
        )
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Graceful shutdown — run session learner before teardown
        try:
            await asyncio.wait_for(
                session_learner.run_shutdown_learning(), timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Session learning timed out (30s limit).")
        except Exception as exc:
            logger.error("Session learning failed: %s", exc)
        await memory_auditor.stop()
        voice_listener.stop()
        file_agent.stop_watching()
        web_agent.close()
        if mcp_client.enabled:
            await mcp_client.disconnect_all()
        await llm_core.close()
        ui_layer.stop()
        logger.info("JARVIS shut down gracefully.")


if __name__ == "__main__":
    # Use SelectorEventLoop on Windows to avoid ProactorEventLoop subprocess
    # cleanup noise ("Event loop is closed" warnings on shutdown)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJARVIS shut down.")
