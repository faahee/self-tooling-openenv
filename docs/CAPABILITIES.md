# JARVIS — Complete Capabilities Documentation

Everything JARVIS can do, organized by agent and module.

---

## 1. OS AGENT (`agents/os_agent.py`)

### App Launching
| Say | What happens |
|-----|-------------|
| "open chrome" / "launch spotify" / "start discord" | Opens app from 257 discovered Start Menu apps |
| "open notepad" / "open vscode" / "open terminal" | Matches against alias dictionary |

**All App Aliases:**
```
chrome, google chrome, firefox, mozilla firefox, edge, microsoft edge,
vscode, visual studio code, notepad, terminal, windows terminal,
powershell, cmd, command prompt, explorer, file explorer, task manager,
discord, slack, telegram, whatsapp, zoom, teams, microsoft teams,
spotify, obs, vlc, steam, photos, camera, maps, weather, news,
word, excel, powerpoint, paint, calculator, brave, settings, mail, store, clock
```

### App / Process Control
| Say | What happens |
|-----|-------------|
| "kill chrome" / "close spotify" | Kills matching process |
| "screenshot" | Saves screenshot to `data/screenshots/` |
| "volume 50" | Sets system volume to 50% |
| "cpu" / "processes" / "what's eating RAM" | Lists top 10 processes by CPU |
| "gpu" / "vram" | Shows GPU name, VRAM used/total, load% |
| "windows" | Lists all open windows |
| "run [command]" | Executes a shell command |

### GUI Automation — Messaging
| Say | What happens |
|-----|-------------|
| "send hello to john on whatsapp" | Opens WhatsApp, searches contact, sends message |
| "message rifana on telegram saying hi" | Opens Telegram, searches contact, sends message |
| "tell ali good morning on whatsapp" | Same as above |
| "dm sarah meeting at 5 on telegram" | Same as above |

**Supported patterns:** `send/message/tell/dm/text [message] to [contact] on [app]`

### GUI Automation — Browser
| Say | What happens |
|-----|-------------|
| "open chrome and go to youtube.com" | Opens Chrome, navigates to URL |
| "open brave and search python tutorials" | Opens Brave, searches on Google |
| "open edge and visit github.com" | Opens Edge, navigates to URL |

**Supported browsers:** chrome, brave, edge, firefox

### GUI Automation — Open + Type
| Say | What happens |
|-----|-------------|
| "open notepad and type hello world" | Opens Notepad, types "hello world" |
| "open word and write meeting notes" | Opens Word, types "meeting notes" |

**Pattern:** `open [app] and type/write/enter [text]`

### System Info
| Say | What happens |
|-----|-------------|
| "how much RAM am I using" | Returns CPU%, RAM (used/total), GPU, top processes, disk |
| "what is my GPU temperature" | Returns GPU name, VRAM, load% |

---

## 2. TERMINAL AGENT (`agents/terminal_agent.py`)

### Direct Commands (Instant — No LLM)
These execute immediately via pattern matching:

**Navigation:**
| Command | What happens |
|---------|-------------|
| `cd [path]` | Changes working directory (supports `~`, `-`, relative/absolute) |
| `ls` / `dir` / `ll` | Lists current directory |
| `pwd` / `cwd` | Shows working directory |

**Package Managers (passthrough):**
| Command | What happens |
|---------|-------------|
| `pip install flask` | Runs pip directly |
| `npm install express` | Runs npm directly |
| `yarn add react` | Runs yarn directly |
| `conda install numpy` | Runs conda directly |
| `install requests` | Shortcut → `pip install requests` |

Also: `pip3`, `pipx`, `poetry`, `uv`, `npx`, `pnpm`, `bun`, `mamba`

**Git (passthrough):**
| Command | What happens |
|---------|-------------|
| `git status` | Runs git directly |
| `git commit -m "msg"` | Runs git directly |
| `git push` / `git pull` / `git clone` | Runs git directly |

**Language Runtimes:**
| Command | What happens |
|---------|-------------|
| `python main.py` | Runs Python |
| `node server.js` | Runs Node.js |
| `cargo build` / `go run main.go` | Runs Rust/Go |
| `dotnet run` / `mvn package` / `gradle build` | Runs .NET/Java |

**Build & Test:**
| Command | What happens |
|---------|-------------|
| `pytest` / `jest` / `mocha` | Runs test framework |
| `make` / `cmake` / `tox` / `nox` | Runs build tools |

**Docker:**
| Command | What happens |
|---------|-------------|
| `docker build .` | Runs Docker directly |
| `docker-compose up` | Runs docker-compose directly |
| `podman run` | Runs Podman directly |

**File Reading:**
| Command | What happens |
|---------|-------------|
| `cat file.py` / `type readme.md` | Reads and displays file with line numbers |

**Network:**
| Command | What happens |
|---------|-------------|
| `ping google.com` | Runs ping |
| `curl https://api.example.com` | Runs curl |
| `ssh user@host` | Runs ssh |

Also: `ipconfig`, `netstat`, `nslookup`, `tracert`, `wget`, `scp`

**Directory Creation:**
| Command | What happens |
|---------|-------------|
| `mkdir my-project` | Creates directory |

### LLM Agent Loop (Complex Tasks)
For anything that isn't a direct command, JARVIS uses an LLM planning loop:

| Say | What happens |
|-----|-------------|
| "create a python file that prints hello world" | LLM generates command → executes → checks done |
| "create a flask project with routes" | Multi-step: creates files, installs dependencies |
| "list all python files" | LLM generates `Get-ChildItem -Filter *.py` |
| "fix the indentation in this file" | LLM generates autopep8/black commands |

**How it works:**
1. Sends task to phi3:mini LLM
2. LLM responds with a command in a code block
3. Runs command via PowerShell
4. Feeds output back to LLM
5. Repeats until LLM says "DONE" (max 10 steps)

**Safety:** These commands are BLOCKED:
```
format C:, del /s /f, rm -rf /, rmdir /s /q C:\
reg delete, bcdedit, diskpart, shutdown, sfc, dism
```

---

## 3. FILE AGENT (`agents/file_agent.py`)

| Say | What happens |
|-----|-------------|
| "find my train.py file" | Searches by filename across disks |
| "search for config files" | Glob search + content search |
| "where is my resume" | Searches by filename |
| "read the PDF in Downloads" | Finds file → extracts text (PDF/DOCX/text) |
| "summarize my Downloads folder" | Shows: file count, size, types, largest files, subdirs |
| "find duplicate files on desktop" | MD5 hash comparison, shows duplicate groups |

**Supports reading:** `.txt`, `.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`, `.html`, `.css`, `.csv`, `.xml`, `.ini`, `.cfg`, `.toml`, `.java`, `.cpp`, `.c`, `.h`, `.rs`, `.go`, `.sh`, `.bat`, `.pdf`, `.docx`, `.log`, `.env`, `.gitignore`, `.dockerfile`

**Content search:** Searches INSIDE files for text matches (line number + context)

**File watching:** Monitors `C:/Users` and `C:/jarvis` for real-time changes

---

## 4. IMAGE ANALYZER (`agents/image_analyzer.py`)

| Say | What happens |
|-----|-------------|
| "analyze this screenshot" | Takes screenshot → runs vision model |
| "what is in this image" | Analyzes with moondream (fast, 1.8GB) |
| "read the text in this image" | OCR via vision model |
| "describe what you see on screen" | Screenshot + analysis |

**Two vision models:**
- **Fast:** `moondream` (1.8GB VRAM) — default for quick analysis
- **Deep:** `llava:7b` (3.9GB VRAM) — for complex/detailed analysis

Can analyze: screenshots, image files, clipboard images

---

## 5. MEMORY SYSTEM (`memory/`)

### What JARVIS Remembers

| Say | What happens |
|-----|-------------|
| "remember that I use Python 3.11" | Stores as a fact in knowledge graph + vector index |
| "what did I tell you about my project" | Semantic search across all memories |
| "what are my goals" | Retrieves from personal model + memory |
| "recall our last conversation" | Fetches recent interactions from SQLite |

**Memory types:** interaction, fact, goal, project, preference

**Knowledge Graph:** Stores structured triples (subject → predicate → object)
- Example: "Faheem → uses → Python 3.11"
- Can traverse relationships 2 hops deep

**Vector Index (FAISS):** Semantic similarity search across all stored text

**Personal Model tracks:**
- Your name, goals (long-term + short-term), current projects
- Peak activity hours, session length, preferred detail level
- Communication style, code language preference
- Most used intents, common topics
- Total interaction count

**Interaction Logger (SQLite):** Every conversation saved with timestamp, intent, duration, cache hit status

---

## 6. VOICE INPUT & TTS (`perception/voice_listener.py` + `ui/ui_layer.py`)

### Voice Input
| Action | What happens |
|--------|-------------|
| Say "Jarvis" | Wake word detected, starts recording |
| Speak your command | Records until 1.5s silence |
| | Transcribed via Whisper (base.en, CPU) |

**Auto-selects best mic:** Prioritizes Realme Bluetooth → ASUS AI Noise-Cancelling → Realtek Array → Default

### Text-to-Speech
- Every response is spoken aloud via Windows SAPI5
- Generates WAV → plays via winsound (no conflict with mic recording)
- Speed: 175 WPM, Volume: 90%

---

## 7. LLM ENGINE (`core/llm_core.py`)

| Feature | Details |
|---------|---------|
| Model | `jarvis-phi3` (custom, FROM phi3:mini) |
| Agent model | `phi3:mini` (base, for terminal loop) |
| Context window | 1024 tokens (fits in 4GB VRAM) |
| Max output | 100 tokens (general), 500 tokens (agent loop) |
| Temperature | 0.3 (low hallucination) |
| Streaming | Yes, token-by-token display |
| Backend | Ollama (local, http://localhost:11434) |

---

## 8. INTENT ROUTING (`core/intent_classifier.py` + `core/orchestrator.py`)

JARVIS auto-classifies every input into one of 9 intents:

| Intent | Routes To | Trigger Examples |
|--------|-----------|-----------------|
| `TERMINAL_TASK` | Terminal Agent | pip, npm, git, python, create file, run tests |
| `OS_COMMAND` | OS Agent | open, launch, kill, close, volume, send message |
| `FILE_OP` | File Agent | find file, read PDF, summarize folder, duplicates |
| `IMAGE_ANALYZE` | Image Analyzer | analyze screenshot, what's on screen |
| `MEMORY_QUERY` | Memory System | remember, recall, what did I say |
| `CODE_TASK` | LLM (code mode) | explain code, debug error, write function |
| `SYSTEM_QUERY` | LLM + system state | CPU usage, RAM, GPU temperature |
| `WEB_TASK` | LLM | search for news, weather, documentation |
| `CHAT` | LLM | hello, joke, good morning |

**Classification:** Keyword fast-path (1.0 confidence) → Embedding similarity (all-MiniLM-L6-v2) → Fallback to CHAT if < 0.6

---

## 9. CACHING & PERFORMANCE (`core/response_cache.py`)

- **Exact match cache:** O(1) lookup, 500 entries max
- **Semantic cache:** Embedding similarity (threshold 0.92), 500 entries max
- **System state cache:** Auto-invalidated for CPU/RAM/GPU queries
- **Context prefetch:** Every 30s, pre-builds context for active app
- **Memory flush:** Every 60s, writes pending memories to disk

---

## 10. API ENDPOINTS (`core/orchestrator.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/input` | Send text input, get JARVIS response |
| GET | `/status` | Get system state (CPU, RAM, GPU, processes) |
| GET | `/health` | Health check |
| POST | `/memory/add` | Add a memory (text + category) |
| GET | `/memory/query?q=...` | Query memories semantically |

---

## Quick Reference — What To Say

```
APPS:        "open chrome" / "launch spotify" / "kill discord"
MESSAGES:    "send hi to john on whatsapp"
BROWSE:      "open brave and go to github.com"
TYPE:        "open notepad and type hello"
TERMINAL:    "pip install flask" / "git status" / "python main.py"
COMPLEX:     "create a react app" / "fix indentation" / "list all py files"
FILES:       "find my resume" / "read the PDF" / "summarize Downloads"
IMAGES:      "analyze this screenshot" / "what's on screen"
MEMORY:      "remember I use Python 3.11" / "what are my goals"
SYSTEM:      "cpu usage" / "gpu info" / "volume 50"
VOICE:       Just say "Jarvis" + your command
```
