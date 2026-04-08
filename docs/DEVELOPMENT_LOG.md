# JARVIS Development Log

## Project Overview

**JARVIS** — A local, self-evolving AI OS agent built in Python, powered by Llama 3.2 3B via Ollama.

- **Runtime**: Python 3.13, Windows 11, ASUS ROG Strix G713RC, RTX 3050 4GB VRAM
- **LLM**: Llama 3.2 3B via Ollama (`jarvis-llama`), all-MiniLM-L6-v2 embedder on CPU
- **Server**: FastAPI/Uvicorn on 127.0.0.1:8000
- **Workspace**: `E:\projects\jarvis` with `.venv` virtual environment
- **Databases**: `data/brain.db` (tool registry), `data/interactions.db` (interaction log)
- **JSON stores**: `data/knowledge_graph.json`, `data/indexed_texts.json`, `data/personal_model.json`

---

## Phase 1: Self-Evolving Brain Foundation

Built the core self-evolving brain for the JARVIS AI OS agent:

- Implemented a **6-step reasoning pipeline** in `DecisionEngine`
- Fixed intent routing and agent re-routing
- Fixed battery handler, LLM hallucination issues, datetime injection
- Fixed media control ordering and PDF tool synthesis/code validator
- All 6/6 tests passing at end of phase

---

## Phase 2: JSON Decision Engine Rewrite

**Problem**: The original decision engine used fragile natural language output parsed with regex, leading to unreliable behavior.

**Solution**: Replaced with strict JSON output format.

### Changes Made

- **`brain/decision_engine.py`**:
  - Rewrote `_REASONING_SYSTEM_PROMPT` to instruct the LLM to output strict JSON
  - Rewrote `_run_reasoning()` with retry logic
  - Rewrote `_parse_reasoning()` using `json.loads()` with field validation
  - Rewrote `handle()` and `_display_reasoning()` and `_reasoning_to_plan()`
  - Added `_VALID_AGENTS` class variable for agent name validation
  - Auto-fixes invalid agent names, fallback dict on parse failure

- **`test_brain.py`**:
  - Updated Test 2 and Test 5 mocks to return proper JSON reasoning format

**Result**: All 6/6 tests passing.

---

## Phase 3: Code Validator Rewrite (4-Level Risk Scoring)

**Problem**: Binary allow/block import system was a game of whack-a-mole — every new module needed manual classification.

**Solution**: Replaced with a 4-level risk scoring system using full AST analysis.

### Risk Levels

| Level | Category | Action |
|-------|----------|--------|
| 0 | Safe stdlib (math, json, datetime, etc.) | Auto-allow |
| 1 | Data/utility libraries + logging | Allow with log |
| 2 | System access (os, subprocess, shutil, etc.) | Require human approval |
| 3 | Network/dangerous (socket, ctypes, importlib) | Always block |

### Changes Made

- **`brain/code_validator.py`** (full rewrite):
  - `ValidationResult` dataclass for structured results
  - `validate()` synchronous API + `validate_async()` with interactive Level 2 approval
  - 6-step AST pipeline:
    1. Syntax check
    2. Import analysis
    3. Dangerous call detection (`_DANGEROUS_CALLS` frozenset)
    4. Dangerous builtin detection (`_DANGEROUS_BUILTINS` frozenset)
    5. Pattern matching (attribute writes via `_BLOCKED_ATTR_WRITES`)
    6. Structure validation
  - Backup saved at `brain/code_validator.py.bak`

- **`config.yaml`**:
  - Added `validator:` section with `risk_levels:` containing level_0 through level_3 module lists

- **`ui/ui_layer.py`**:
  - Added `prompt_import_approval()` for Level 2 interactive approval prompts

- **`brain/tool_synthesizer.py`**:
  - Updated to use new `ValidationResult` interface

- **`main.py`**:
  - Updated validator initialization

**Result**: All 6/6 tests passing.

---

## Phase 4: Memory Auditor

**Problem**: Memory stores (knowledge graph, indexed texts, tool registry, interaction logs) accumulate garbage over time — ghost entries, hallucination artifacts, dead references, orphan files.

**Solution**: Built `brain/memory_auditor.py` with 5 automated garbage collection checks.

### 5 Audit Checks

1. **Tool Health**: Deactivate tools with low success rate, archive unused tools, remove ghost entries, clean old tool versions
2. **Knowledge Graph**: Remove dead references, quarantine hallucination artifacts, deduplicate entries
3. **Indexed Texts**: Clean empty entries, remove hallucination artifacts, purge entries older than 90 days
4. **Generated Tools Directory**: Delete orphan `.py` files not in registry, move deactivated tools to `archived/`
5. **Interaction Logs**: Compress entries older than 7 days into weekly summaries, keep last 1000 entries

### Changes Made

- **`brain/memory_auditor.py`** (new file):
  - `MemoryAuditor` class
  - `AuditReport` dataclass with `.summary` property
  - Backup/restore helpers for safe JSON modifications
  - Runs on startup, every 30 minutes (background task), and on-demand

- **`main.py`**:
  - Section 11c: `MemoryAuditor` init + `await memory_auditor.start()`
  - Attached to orchestrator as `memory_auditor` attribute
  - `memory_auditor.stop()` in finally block

- **`core/orchestrator.py`**:
  - On-demand trigger: phrases like "clean your memory" / "memory audit" bypass intent classification and call auditor directly

- **`ui/ui_layer.py`**:
  - Added `show_audit_report()` — Rich formatted memory audit results panel

### Live Test Results

On first startup scan:
- Removed 1 ghost entry (`date_display_tool`)
- Cleaned 11 entries total
- Freed 2.2 KB

---

## Phase 5: Session Learner

**Problem**: JARVIS loses all learned context when a session ends — mistakes repeat, successful patterns aren't reinforced.

**Solution**: Built `brain/session_learner.py` that extracts cross-session learning patterns and persists them for future use.

### 4 Pattern Types

| Type | Description | Detection |
|------|-------------|-----------|
| A | Repeated failures | Same intent failed 2+ times then succeeded |
| B | Successful new tools | Tools created this session with success_rate >= 0.7 |
| C | Intent misclassification | User correction detected after wrong classification |
| D | Agent routing | Re-route events found in tool event details |

### Changes Made

- **`brain/session_learner.py`** (new file):
  - `SessionLearner` class with `run_shutdown_learning()` and `get_lessons_learned()`
  - `_write_lessons()`: Persists to `personal_model.json` `learned_patterns` section
    - Reinforces duplicate patterns (increments count)
    - Resolves conflicts by confidence score
    - Auto-promotes to confidence=1.0 after 3 reinforcements
  - `SessionSummary` dataclass with `.format()` method
  - `load_lessons_for_context()`: Static method for context_builder injection
    - Max 5 lessons, confidence > 0.7, skips deactivated tool references
  - Saves session summaries to `data/session_summaries/[datetime].txt`

- **`core/context_builder.py`**:
  - Added `Path` import
  - `_load_lessons()` static method called once at `__init__`
  - Injects up to 5 high-confidence lessons into system prompt as:
    ```
    Learned from past sessions:
    - lesson1
    - lesson2
    ```

- **`core/orchestrator.py`**:
  - On-demand trigger: "what did you learn" / "lessons learned" phrases bypass intent classification

- **`main.py`**:
  - Section 11d: `SessionLearner` init
  - Attached to orchestrator as `session_learner` attribute
  - Shutdown: `await asyncio.wait_for(session_learner.run_shutdown_learning(), timeout=30.0)`

- **`ui/ui_layer.py`**:
  - Added `show_session_summary()` — Rich Panel on shutdown showing duration, task counts, top lesson

**Result**: All 6/6 tests passing.

---

## Phase 6: Live Testing

Ran JARVIS live with `python main.py`:

- Booted successfully with all 17 modules
- Memory Auditor ran startup scan automatically
- Tested weather query: "what is weather in chennai" → `API_TASK` (confidence 1.00) → correct response
- Tested follow-up: "where you got this information" → `CHAT` (confidence 0.48) → correct conversational response
- All systems operational

---

## Project Structure

```
E:\projects\jarvis\
├── main.py                          # Entry point, module init, shutdown hooks
├── config.yaml                      # Configuration (LLM, validator risk levels, etc.)
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── Modelfile                        # Ollama model definition
├── README.md                        # Project readme
├── CAPABILITIES.md                  # Capability documentation
│
├── brain/                           # Self-evolving brain
│   ├── __init__.py
│   ├── decision_engine.py           # 6-step JSON reasoning pipeline
│   ├── code_validator.py            # 4-level risk scoring AST validator
│   ├── memory_auditor.py            # 5-check garbage collection system
│   ├── session_learner.py           # Cross-session pattern learning
│   ├── tool_registry.py             # Tool registration and management
│   ├── tool_synthesizer.py          # Runtime tool generation
│   ├── tool_loader.py               # Tool loading utilities
│   ├── sandbox_executor.py          # Sandboxed code execution
│   └── tools/
│       ├── __init__.py
│       ├── builtin/                 # Built-in tools
│       ├── generated/               # Runtime-generated tools
│       └── archived/                # Deactivated tools archive
│
├── agents/                          # 10 specialized agents
│   ├── api_agent.py                 # External API calls
│   ├── data_agent.py                # Data processing
│   ├── email_agent.py               # Email operations
│   ├── file_agent.py                # File system operations
│   ├── image_analyzer.py            # Image analysis
│   ├── os_agent.py                  # OS-level operations
│   ├── scheduler_agent.py           # Task scheduling
│   ├── terminal_agent.py            # Terminal command execution
│   ├── web_agent.py                 # Web browsing/scraping
│   └── workflow_agent.py            # Multi-step workflow orchestration
│
├── core/                            # Core orchestration
│   ├── orchestrator.py              # Main request router
│   ├── llm_core.py                  # LLM interface (Ollama)
│   ├── intent_classifier.py         # Intent classification
│   ├── context_builder.py           # System prompt + lesson injection
│   ├── response_cache.py            # Response caching
│   └── checkpoint.py                # State checkpointing
│
├── memory/                          # Memory subsystem
│   ├── interaction_logger.py        # SQLite interaction logging
│   ├── memory_system.py             # FAISS vector memory
│   └── personal_model.py            # User preference modeling
│
├── perception/                      # Input perception
│   └── voice_listener.py            # Voice input
│
├── ui/                              # User interface
│   └── ui_layer.py                  # Rich terminal UI
│
├── data/                            # Data stores
│   ├── brain.db                     # Tool registry (SQLite)
│   ├── interactions.db              # Interaction log (SQLite)
│   ├── knowledge_graph.json         # Knowledge graph
│   ├── indexed_texts.json           # Indexed text snippets
│   ├── personal_model.json          # Personal model + learned patterns
│   ├── faiss_index.bin              # FAISS vector index
│   ├── file_index.json              # File index
│   ├── session_summaries/           # Session learning summaries
│   └── screenshots/                 # Screenshot storage
│
├── tools/                           # Standalone utilities
│   └── clean_memory.py              # Manual memory cleanup script
│
└── test_brain.py                    # Brain test suite (6 tests)
```

---

## Testing Notes

- **Test runner**: `python test_brain.py` (NOT `python -m pytest` — tests use `asyncio.run()`)
- **Exit code 1** from stderr warnings (TensorFlow) is normal — tests still pass
- **Encoding**: Use `$env:PYTHONIOENCODING = "utf-8"` in PowerShell for clean output
- **All 6/6 tests passing** as of latest run

---

## Bug Fixes & Lessons Learned

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Intent routing failures | Regex parsing of LLM output | Switched to strict JSON output |
| LLM hallucination in reasoning | Free-form text output | Constrained to JSON schema with field validation |
| Import validation whack-a-mole | Binary allow/block lists | 4-level risk scoring with AST analysis |
| Ghost tool entries | Tools deleted from disk but not registry | Memory auditor removes ghost entries on startup |
| BOM encoding in files | PowerShell heredoc added UTF-8 BOM | Stripped with `[System.IO.File]::ReadAllText` |
| Session knowledge loss | No persistence between sessions | Session learner extracts and persists patterns |
| Datetime injection errors | Missing datetime in context | Fixed datetime injection in context builder |
| Media control ordering | Wrong priority in OS agent | Fixed ordering logic |
