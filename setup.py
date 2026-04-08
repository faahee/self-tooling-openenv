"""JARVIS Setup Script — First-run initialization.

Checks prerequisites, pulls Ollama models, creates data directories,
initializes databases, and runs a health check.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REQUIRED_DIRS = [DATA_DIR, SCREENSHOTS_DIR]

PERSONAL_MODEL_PATH = DATA_DIR / "personal_model.json"
KNOWLEDGE_GRAPH_PATH = DATA_DIR / "knowledge_graph.json"
DB_PATH = DATA_DIR / "interactions.db"
FILE_INDEX_PATH = DATA_DIR / "file_index.json"

DEFAULT_PERSONAL_MODEL = {
    "name": "Jarvis",
    "goals": {
        "long_term": [
            "Build transformative deep tech — AI OS agent, autonomous systems",
            "Build VALLUGE fashion AI startup for Indian market",
            "Master ML/AI from fundamentals to deployment",
        ],
        "current_projects": ["JARVIS OS agent", "VALLUGE", "ML study plan"],
        "short_term": [],
    },
    "cognitive_patterns": {
        "peak_hours": [],
        "avg_session_length_mins": 0,
        "preferred_detail_level": "detailed",
        "learning_style": "examples",
    },
    "preferences": {
        "communication_style": "direct",
        "tts_enabled": True,
        "proactive_suggestions": True,
        "code_language": "python",
    },
    "interaction_patterns": {
        "total_interactions": 0,
        "most_used_intents": {},
        "common_topics": [],
        "last_active": None,
    },
    "emotional_patterns": {
        "stress_indicators": [],
        "energized_indicators": [],
    },
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    user_input  TEXT NOT NULL,
    intent      TEXT,
    response    TEXT NOT NULL,
    duration_ms INTEGER,
    cache_hit   INTEGER DEFAULT 0,
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_intent ON interactions(intent);
"""


def _print_header() -> None:
    print("\n" + "=" * 60)
    print("  JARVIS — First-Run Setup")
    print("=" * 60 + "\n")


def check_python_version() -> bool:
    """Verify Python >= 3.11."""
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major, minor) >= (3, 11)
    tag = "OK" if ok else "FAIL"
    print(f"[{tag}] Python version: {major}.{minor}")
    if not ok:
        print("      JARVIS requires Python 3.11 or newer.")
    return ok


def check_nvidia_gpu() -> bool:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"[OK]   NVIDIA GPU detected: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass
    print("[WARN] No NVIDIA GPU detected. JARVIS needs an NVIDIA GPU for local LLM inference.")
    return False


def create_directories() -> None:
    """Create data directories."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print("[OK]   Data directories created.")


def check_ollama_installed() -> bool:
    """Check if Ollama binary is available."""
    if shutil.which("ollama") is not None:
        print("[OK]   Ollama is installed.")
        return True
    print("[FAIL] Ollama is not installed.")
    print("       Download from: https://ollama.ai/download/windows")
    return False


def check_ollama_running() -> bool:
    """Check if Ollama server is listening."""
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("[OK]   Ollama server is running.")
                return True
    except Exception:
        pass
    print("[WARN] Ollama server is not running. Start it with: ollama serve")
    return False


def pull_model(model_name: str) -> bool:
    """Pull an Ollama model."""
    print(f"       Pulling model: {model_name} ...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False, timeout=600,
        )
        if result.returncode == 0:
            print(f"[OK]   Model pulled: {model_name}")
            return True
    except Exception as exc:
        print(f"[FAIL] Could not pull {model_name}: {exc}")
    return False


def create_jarvis_model() -> bool:
    """Create the jarvis-phi3 model from Modelfile."""
    modelfile = Path("Modelfile")
    if not modelfile.exists():
        print("[FAIL] Modelfile not found in project root.")
        return False
    print("       Creating jarvis-phi3 model from Modelfile ...")
    try:
        result = subprocess.run(
            ["ollama", "create", "jarvis-phi3", "-f", str(modelfile)],
            capture_output=False, timeout=300,
        )
        if result.returncode == 0:
            print("[OK]   jarvis-phi3 model created.")
            return True
    except Exception as exc:
        print(f"[FAIL] Could not create jarvis-phi3: {exc}")
    return False


def set_env_variables() -> None:
    """Set Ollama optimization env vars for Windows (user scope)."""
    env_vars = {
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_NUM_PARALLEL": "1",
    }
    if platform.system() == "Windows":
        for key, val in env_vars.items():
            try:
                subprocess.run(
                    ["setx", key, val],
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass
        print("[OK]   Windows environment variables set (OLLAMA_KEEP_ALIVE, OLLAMA_FLASH_ATTENTION, OLLAMA_NUM_PARALLEL).")
    else:
        print("[SKIP] Environment variable setup is Windows-only.")


def create_personal_model() -> None:
    """Write default personal model JSON."""
    if not PERSONAL_MODEL_PATH.exists():
        PERSONAL_MODEL_PATH.write_text(json.dumps(DEFAULT_PERSONAL_MODEL, indent=2), encoding="utf-8")
        print("[OK]   data/personal_model.json created.")
    else:
        print("[SKIP] data/personal_model.json already exists.")


def create_knowledge_graph() -> None:
    """Write empty knowledge graph JSON."""
    if not KNOWLEDGE_GRAPH_PATH.exists():
        KNOWLEDGE_GRAPH_PATH.write_text(json.dumps({"nodes": [], "links": []}, indent=2), encoding="utf-8")
        print("[OK]   data/knowledge_graph.json created.")
    else:
        print("[SKIP] data/knowledge_graph.json already exists.")


def create_file_index() -> None:
    """Write empty file index JSON."""
    if not FILE_INDEX_PATH.exists():
        FILE_INDEX_PATH.write_text(json.dumps({}, indent=2), encoding="utf-8")
        print("[OK]   data/file_index.json created.")
    else:
        print("[SKIP] data/file_index.json already exists.")


def init_database() -> None:
    """Initialize SQLite database with interactions table."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(CREATE_TABLE_SQL)
    conn.close()
    print("[OK]   SQLite database initialized.")


def run_health_check() -> None:
    """Send a test prompt to jarvis-phi3 and measure response time."""
    print("\n       Running health check ...")
    try:
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "model": "jarvis-phi3",
            "prompt": "Say hello in one sentence.",
            "stream": False,
            "options": {"num_predict": 30, "num_gpu": 99},
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start

        response_text = body.get("response", "")
        print(f"[OK]   Health check passed in {elapsed:.1f}s")
        print(f"       Response: {response_text.strip()[:120]}")
    except Exception as exc:
        print(f"[WARN] Health check failed: {exc}")
        print("       Make sure Ollama is running and jarvis-phi3 model is created.")


def main() -> None:
    _print_header()

    py_ok = check_python_version()
    if not py_ok:
        sys.exit(1)

    check_nvidia_gpu()
    create_directories()

    ollama_installed = check_ollama_installed()
    if not ollama_installed:
        print("\n[!] Install Ollama first, then re-run this setup.")
        sys.exit(1)

    ollama_running = check_ollama_running()

    if ollama_running:
        pull_model("phi3:mini")
        pull_model("moondream")
        create_jarvis_model()

    set_env_variables()
    create_personal_model()
    create_knowledge_graph()
    create_file_index()
    init_database()

    if ollama_running:
        run_health_check()

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print("\n  Next steps:")
    print("    1. Make sure Ollama is running:  ollama serve")
    print("    2. Start JARVIS:                 python main.py")
    print()


if __name__ == "__main__":
    main()
