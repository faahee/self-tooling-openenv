"""Intent Classifier — Fast pre-LLM routing using sentence-transformers."""
from __future__ import annotations

import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user input into one of 8 intent categories in under 100ms.

    Uses cosine similarity against pre-embedded example phrases per category.
    Runs on CPU with the tiny all-MiniLM-L6-v2 model (~80MB).
    """

    INTENT_LABELS = [
        "TERMINAL_TASK",
        "OS_COMMAND",
        "FILE_OP",
        "IMAGE_ANALYZE",
        "MEMORY_QUERY",
        "CODE_TASK",
        "WEB_TASK",
        "SYSTEM_QUERY",
        "DATA_TASK",
        "EMAIL_TASK",
        "SCHEDULE_TASK",
        "API_TASK",
        "WORKFLOW",
        "CHAT",
    ]

    INTENT_EXAMPLES: dict[str, list[str]] = {
        "TERMINAL_TASK": [
            "run pip install numpy",
            "install flask",
            "pip install requests",
            "npm install express",
            "git init",
            "git commit",
            "git push",
            "git clone a repository",
            "create a python file",
            "create a flask project",
            "create a new project folder",
            "run python main.py",
            "run the tests",
            "run my python script",
            "execute the build script",
            "mkdir src",
            "cd into the project folder",
            "show me the files in this directory",
            "edit main.py and add error handling",
            "fix the bug in parser.py",
            "install the requirements",
            "build the project",
            "create a react app",
            "set up a virtual environment",
            "run pytest",
            "docker build this project",
            "compile the code",
            "add a new route to the server",
            "initialize a new node project",
        ],
        "OS_COMMAND": [
            "open VS Code",
            "kill chrome",
            "set volume to 50 percent",
            "close all windows",
            "minimize all windows",
            "minimize all screens",
            "show desktop",
            "hide all windows",
            "open whatsapp",
            "launch spotify",
            "open telegram and send a message",
            "start discord",
            "open calculator",
            "open brave",
            "open notepad",
            "open chrome",
            "open file explorer",
            "launch obs",
            "open steam",
            "start zoom",
            "open settings",
            "take a screenshot",
            "close discord",
            "kill spotify",
            "open vlc",
            # Media / music playback
            "play some music",
            "play songs",
            "play karuppu songs",
            "play suriya movie songs",
            "play weekend songs",
            "play tamil songs",
            "play my playlist",
            "play a song",
            "play music on youtube",
            "play blinding lights",
            "play some hip hop",
            "play relaxing music",
            "play lo-fi beats",
            "play bollywood songs",
            "play rock music",
            "pause the music",
            "stop the music",
            "next song",
            "previous song",
            "skip this song",
            "open whatsapp and send hi to rifana",
            "send hello to john on whatsapp",
            "message rifana on telegram",
            "send a message to sarah on whatsapp saying hey",
            "tell mike hello on telegram",
            "open chrome and go to youtube.com",
            "open brave and search for python tutorials",
            "find where vs code is located",
            "where is python installed",
            "locate the chrome executable",
            "where is node installed",
            "find the path of git",
            "where is ffmpeg",
            "locate vs code",
            "search for rifana in whatsapp",
            "find john in telegram",
            "look up sarah in whatsapp",
            "open chat with mike on telegram",
            # GUI automation — type/write/click in apps
            "type hello in notepad",
            "write hello world in notepad",
            "type something in wordpad",
            "write text in notepad",
            "open notepad and type hello",
            "launch notepad and type something",
            "type i love you in notepad",
            "in notepad type my name",
            "click the button in notepad",
            "press enter in notepad",
            "input text in wordpad",
            "keyboard input in notepad",
        ],
        "FILE_OP": [
            "find my train.py file",
            "read the PDF in Downloads",
            "search for config files",
            "summarize my Downloads folder",
            "create a folder on desktop",
            "create a folder named kool on desktop",
            "make a new directory in downloads",
            "move this folder to desktop",
            "rename this folder",
            "find duplicate files on desktop",
            "where are my pictures",
            "where my pics on e drive",
            "find my photos on d drive",
            "where are my videos",
            "show me my images on e drive",
            "find all jpg files",
            "where is my document",
            "locate my music files",
            "find screenshots on desktop",
            "where my files at",
            "show my photos",
            "find images on e drive",
            "search for mp4 files",
            "where are my downloads",
            "find pdf files in documents",
            "locate my excel files",
            "save this to a file",
            "save the report as a document",
            "save as a Word document",
            "save results to a folder",
            "create a new document",
            "create a new text file",
            "save to a new folder named AI News",
            "save search results to a folder",
            "copy and paste into notepad",
            "write this text to a file",
            "save file as PDF",
            "rename this file",
            "move file to folder",
            "save invoices to folder",
        ],
        "IMAGE_ANALYZE": [
            "analyze this screenshot",
            "what is in this image",
            "read the text in this image",
            "what error is on my screen",
            "describe what you see on screen",
            "can you see my screen",
            "look at my screen",
            "what's on my screen",
            "what software is this",
            "which software is that",
            "which app is this",
            "what is on my screen right now",
            "what am i looking at",
            "take a look at my screen",
            "identify this application",
        ],
        "MEMORY_QUERY": [
            "what did I tell you about VALLUGE",
            "what are my goals",
            "remember that I use Python 3.11",
            "what do you know about me",
            "recall our last conversation about ML",
        ],
        "CODE_TASK": [
            "explain this code",
            "debug this error",
            "write a function for sorting",
            "refactor this python script",
            "what does this regex do",
        ],
        "WEB_TASK": [
            "search for latest AI news",
            "what is the weather today",
            "find information about PyTorch 2.0",
            "look up the documentation for FastAPI",
            "search the web for CUDA installation guide",
            "google python tutorials",
            "search for best laptops 2024",
            "look up how to use docker",
            "what is the latest version of react",
            "find news about openai",
            "fetch this url https://example.com",
            "open this webpage",
            "what is machine learning",
            "who invented python",
            "tell me about transformers in AI",
            "scrape this website",
            "extract all links from this page",
            "scrape tables from the URL",
            "crawl this webpage and get emails",
            "get all prices from this site",
            "harvest data from this page",
            "scrape and save to csv",
            # Current events / news
            "news about iran vs israel",
            "latest news about ukraine war",
            "what is happening in gaza",
            "news today about politics",
            "what happened in the news today",
            "breaking news about stock market",
            "current situation in middle east",
            "weather forecast for tomorrow",
            "live cricket score",
            "bitcoin price today",
            "who won the election",
            "latest updates on covid",
        ],
        "SYSTEM_QUERY": [
            "how much RAM am I using",
            "what is my GPU temperature",
            "what processes are running",
            "how much disk space do I have left",
            "what is my CPU usage right now",
        ],
        "DATA_TASK": [
            "load my sales.csv file",
            "analyze this dataset",
            "summarize the CSV",
            "plot a histogram of age",
            "create a bar chart of revenue",
            "export data to Excel",
            "clean the data and remove nulls",
            "filter rows where age is greater than 30",
            "group by category and sum revenue",
            "generate a data analysis report",
            "show me the top 10 rows",
            "convert JSON to CSV",
            "load data from report.xlsx",
            "what is the average salary in this dataset",
        ],
        "EMAIL_TASK": [
            "send an email to john@example.com",
            "email rifana saying the meeting is at 3pm",
            "check my inbox",
            "read my unread emails",
            "summarize my emails",
            "reply to the last email",
            "search for emails about invoice",
            "compose email to boss",
            "send mail to team",
            "check if I have any new emails",
            "send the report as an email attachment",
            "email the results to me",
            "send this as an email",
            "email me the report",
            "notify me by email",
            "send email attachment to boss",
            "forward this to my email",
        ],
        "SCHEDULE_TASK": [
            "remind me at 3pm to call john",
            "schedule a task every 30 minutes",
            "remind me in 1 hour",
            "run the backup every day at midnight",
            "set a reminder for tomorrow morning",
            "schedule email check every hour",
            "list my scheduled tasks",
            "cancel the scheduled task",
            "remind me to take a break every 2 hours",
            "run this script every Monday at 9am",
        ],
        "API_TASK": [
            "what is the weather in dubai",
            "bitcoin price now",
            "ethereum price in dollars",
            "call this API https://api.example.com",
            "get my public IP address",
            "fetch data from the API",
            "post to the webhook",
            "github repo info for facebook/react",
            "tell me a joke",
            "what is the crypto price today",
            "register api myservice url https://api.example.com",
        ],
        "WORKFLOW": [
            "research competitors, scrape their prices, save to Excel, email me",
            "find top AI news, summarize them, and send me a report",
            "check email then save invoices to folder then notify me",
            "search for python jobs, filter by salary, create a spreadsheet",
            "scrape product prices, analyze trends, generate chart and report",
            "download the CSV, clean it, analyze, and plot the results",
            "go step by step: open browser, search for hotels, scrape prices",
            "collect data from the website then save and analyze it",
            "research the topic, write a report, and email it to me",
            "search the web then summarize then save to file",
            "find news, create a document, write a summary, and email it",
            "scrape data from website, clean it, save to CSV, then email",
            "open browser, search for info, copy results, save to file",
            "first search, then analyze, then create report, then send email",
        ],
        "CHAT": [
            "hello how are you",
            "tell me a joke",
            "what do you think about AI",
            "good morning jarvis",
            "thank you for your help",
            "how are you doing today",
            "what's up",
            "hi jarvis",
            "thanks",
            "that's great",
            "nevermind",
            "ok cool",
        ],
    }

    _SYNONYM_REPLACEMENTS: tuple[tuple[str, str], ...] = (
        ("minimise", "minimize"),
        ("show me the desktop", "show desktop"),
        ("go to desktop", "show desktop"),
        ("hide everything", "show desktop"),
        ("minimize everything", "minimize all windows"),
        ("minimize all screens", "minimize all windows"),
        ("clear the screen", "show desktop"),
        ("open up ", "open "),
        ("launch up ", "launch "),
        ("fire up ", "open "),
        ("start up ", "start "),
        ("turn up volume", "increase volume"),
        ("turn down volume", "decrease volume"),
        ("raise volume", "increase volume"),
        ("lower volume", "decrease volume"),
        ("sound off", "mute"),
        ("sound on", "unmute"),
        ("take screenshot", "take a screenshot"),
        ("capture screen", "take a screenshot"),
        ("screen shot", "screenshot"),
        ("folder name ", "folder named "),
        ("directory name ", "directory named "),
        ("put it in", "move it to"),
        ("put it on", "move it to"),
        ("put this in", "move this to"),
        ("put this on", "move this to"),
        ("create dir", "create directory"),
        ("make dir", "make directory"),
        ("new folder", "create folder"),
        ("new directory", "create directory"),
        ("desktop folder", "desktop"),
        ("downloads folder", "downloads"),
        ("documents folder", "documents"),
        ("open browser", "open chrome"),
        ("start browser", "open chrome"),
        ("play next song", "next song"),
        ("skip to next song", "next song"),
        ("go to next song", "next song"),
        ("go to previous song", "previous song"),
        ("play previous song", "previous song"),
        ("un pause", "resume"),
        ("continue music", "resume music"),
        ("stop the song", "pause music"),
    )

    def __init__(self, config: dict, embedder=None) -> None:
        """Initialize the intent classifier.

        Args:
            config: Full JARVIS configuration dictionary.
            embedder: Optional shared SentenceTransformer instance.
        """
        self.confidence_threshold = config["intent"]["confidence_threshold"]  # 0.6
        if embedder is not None:
            self.model = embedder
        else:
            import torch
            model_name = config["intent"]["classifier_model"]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=device)

        # Pre-embed all example phrases
        self._intent_embeddings: list[tuple[str, np.ndarray]] = []
        for intent, examples in self.INTENT_EXAMPLES.items():
            embeddings = self.model.encode(examples, normalize_embeddings=True, show_progress_bar=False)
            for emb in embeddings:
                self._intent_embeddings.append((intent, emb))

        self._all_embeddings = np.array(
            [e[1] for e in self._intent_embeddings], dtype=np.float32
        )
        logger.info("IntentClassifier loaded with %d example embeddings.", len(self._intent_embeddings))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize common variants so rules and embeddings match better."""
        lower = text.lower().strip()
        for src, dst in IntentClassifier._SYNONYM_REPLACEMENTS:
            lower = lower.replace(src, dst)
        return re.sub(r"\s+", " ", lower).strip()

    # Keywords that must appear at the START of the command (e.g. "pip install numpy").
    # Using startswith-only to avoid false matches like "tell me about python".
    _START_PREFIXES: dict[str, list[str]] = {
        "TERMINAL_TASK": [
            "pip ", "pip3 ", "npm ", "npx ", "yarn ", "pnpm ",
            "conda ", "git ", "docker ",
            "python ", "python3 ", "py ", "node ",
            "pytest", "cargo ", "dotnet ", "mvn ",
            "mkdir ", "cd ", "install ",
            "run python", "run node", "run the ", "run my ",
            "run pytest", "run test", "build the", "compile ",
        ],
        "OS_COMMAND": [
            "open ", "launch ", "start ", "kill ", "close ",
            "volume ", "screenshot", "play ",
            "type ", "write ", "click ", "press ",
            "set volume", "mute ", "unmute ",
            "minimize ", "show desktop",
        ],
    }

    # Phrases that can appear ANYWHERE in the sentence.
    _ANYWHERE_PHRASES: dict[str, list[str]] = {
        "TERMINAL_TASK": [
            "create a python file", "create a script file", "create a code file",
            "create a project", "new project", "create project",
            "edit file", "modify file",
        ],
        "WEB_TASK": [
            "news about", "latest news", "recent news",
            "what happened", "current events", "breaking news",
            "today's news", "war news", "weather today",
            "stock price", "live score", "trending",
        ],
        "FILE_OP": [
            "where my pic", "where my photo", "where my image", "where my video",
            "find my pic", "find my photo", "find my image", "find my video",
            "show my pic", "show my photo",
            "where are my pic", "where are my photo",
            "find all jpg", "find all png", "find all mp4", "find all pdf",
            "duplicate files", "find duplicate",
            "save to file", "save as file", "save to folder", "save to document",
            "save as word", "save as pdf", "save as docx", "write to file",
            "create document", "create a document", "new document",
            "save report", "save results", "save invoices",
            "copy and paste", "paste into",
            "rename file", "move file", "move to folder",
            "create folder", "make folder", "create directory",
            "move folder", "rename folder",
            "where is file", "where is java file", "locate file",
            "find file", "where is document", "where is folder",
        ],
        "DATA_TASK": [
            "load csv", "load excel", "load data", "open csv",
            "analyze data", "plot chart", "bar chart", "histogram",
            "export to excel", "export to csv", "clean data",
            "filter rows", "group by", "sort by",
        ],
        "EMAIL_TASK": [
            "send email", "send mail", "email to", "check inbox",
            "read email", "unread email", "compose email",
            "email me", "email the report", "email attachment",
            "send report", "notify me by email", "forward email",
            "send as attachment", "email it to",
        ],
        "SCHEDULE_TASK": [
            "remind me", "set reminder", "schedule task",
            "every hour", "every day", "every minute",
            "at 3pm", "at 9am", "list scheduled",
        ],
        "API_TASK": [
            "bitcoin price", "ethereum price", "crypto price",
            "weather in", "my public ip", "call api",
            "post to", "github repo",
        ],
        "WORKFLOW": [
            "step by step", "one by one", "then email",
            "and then save", "collect and analyze", "scrape and save",
            "research and report",
        ],
        "OS_COMMAND": [
            "play song", "play music", "play a song", "play my",
            "songs play", "play video", "next song", "previous song",
            "pause music", "stop music", "resume music",
            "minimize all windows", "minimize all screens", "show desktop",
            "hide all windows", "clear desktop",
            # GUI automation phrases
            "type in ", "type into ", "write in ",
            "open and type", "launch and type",
            "type in notepad", "type in wordpad",
            "write in notepad", "write in wordpad",
            "click in ", "press in ",
            "input text", "keyboard input",
            # Process/app listing
            "what apps are running", "running apps", "list processes",
            "running processes", "what is running", "show running",
            "mute the", "unmute the",
        ],
    }

    def classify(self, text: str) -> tuple[str, float]:
        """Classify user input into an intent category.

        Args:
            text: The user's input text.

        Returns:
            Tuple of (intent_label, confidence_score).
            Falls back to "CHAT" if confidence < threshold.
        """
        lower = text.lower().strip()

        # Pre-classifier rule: app name + action verb → force OS_COMMAND
        _GUI_ACTION_VERBS = ("type", "write", "click", "press", "input", "switch", "focus")
        _GUI_APP_NAMES = (
            "notepad", "wordpad", "word", "excel", "paint",
            "chrome", "brave", "edge", "firefox", "vscode",
            "discord", "telegram", "whatsapp", "slack",
            "terminal", "powershell", "cmd", "calculator",
            "spotify", "vlc", "obs", "steam", "teams",
        )
        has_action = any(v in lower for v in _GUI_ACTION_VERBS)
        has_app = any(a in lower for a in _GUI_APP_NAMES)
        if has_action and has_app:
            return "OS_COMMAND", 1.0

        # Pre-classifier rule: desktop/window/media actions → force OS_COMMAND
        _OS_PATTERNS = (
            r"\bminimize all(?: windows)?\b",
            r"\bshow desktop\b",
            r"\bhide all windows\b",
            r"\bclear desktop\b",
            r"\bnext song\b",
            r"\bprevious song\b",
            r"\bpause(?: the)? music\b",
            r"\bresume(?: the)? music\b",
            r"\bmute(?: the)? volume\b",
            r"\bunmute(?: the)? volume\b",
            r"\b(?:increase|decrease)\s+volume\b",
            r"\bturn up volume\b",
            r"\bturn down volume\b",
            r"\braise volume\b",
            r"\blower volume\b",
            r"\b(?:copy|paste|clipboard)\b",
            r"\b(?:notification|notify me|send a notification)\b",
            r"\b(?:powershell command|run powershell|execute powershell)\b",
            r"\b(?:inspect|analyze|observe).+\bscreen\b",
            r"\b(?:switch|focus).+\bwindow\b",
        )
        if any(re.search(pattern, lower) for pattern in _OS_PATTERNS):
            return "OS_COMMAND", 1.0

        # Pre-classifier rule: direct folder/file management → force FILE_OP
        _FILE_RULES = (
            r"\b(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory)\b",
            r"\b(?:move|rename|copy|delete)\s+(?:a\s+)?(?:file|folder|directory)\b",
            r"\b(?:move|put).+\b(?:desktop|downloads|documents)\b",
            r"\b(?:where is|locate|find)\s+.+\b(?:file|files|folder|folders|document|documents)\b",
        )
        if any(re.search(pattern, lower) for pattern in _FILE_RULES):
            return "FILE_OP", 1.0

        # Pre-classifier rule: pure math expressions → force CHAT
        # "what is 2 + 2", "what is 15% of 5000", "calculate 100 / 4"
        _math_rest = re.sub(
            r'^(?:what\s+is\s+|calculate\s+|compute\s+|how\s+much\s+is\s+)', '',
            lower.strip()
        )
        if re.match(r'^[\d\s\+\-\*\/\%\.\(\)^\?]+$', _math_rest):
            return "CHAT", 1.0
        if re.match(r'^\d[\d\s]*\%\s*of\s*[\d\s,]+\??$', _math_rest):
            return "CHAT", 1.0

        # Pre-classifier rule: knowledge / explanation questions → force CHAT
        # These should be answered by the LLM directly, not routed to web search.
        _KNOWLEDGE_PATTERNS = (
            "explain ", "what is a ", "what is an ", "what are ",
            "how does ", "how do ", "how to ", "why does ", "why do ",
            "define ", "meaning of ", "difference between ",
            "what does ", "can you explain", "tell me about ",
            "who invented ", "who created ", "who is the ",
            "what is machine learning", "what is artificial",
            "what is deep learning", "what is neural",
        )
        # Only trigger if no live-data keywords are present
        _LIVE_DATA_WORDS = (
            "weather", "price", "bitcoin", "crypto", "news", "score",
            "stock", "forecast", "ip address", "exchange rate",
        )
        is_knowledge = any(lower.startswith(p) for p in _KNOWLEDGE_PATTERNS)
        has_live_data = any(w in lower for w in _LIVE_DATA_WORDS)
        if is_knowledge and not has_live_data:
            return "CHAT", 1.0

        # Pre-classifier rule: system state queries → force SYSTEM_QUERY
        _SYSTEM_PATTERNS = (
            "my battery", "battery percentage", "battery level", "battery percent",
            "my cpu", "cpu usage", "cpu percent", "processor usage",
            "my ram", "ram usage", "ram used", "memory usage", "how much ram",
            "my gpu", "gpu usage", "gpu load", "vram usage",
            "disk space", "disk usage", "free space",
            "space left", "space available", "space remaining", "drive space",
        )
        if any(p in lower for p in _SYSTEM_PATTERNS):
            return "SYSTEM_QUERY", 1.0

        # "how much space" + drive reference → SYSTEM_QUERY (not FILE_OP)
        # Also catch "how much left in X drive" / "how much i left in X drive"
        if "drive" in lower and ("how much" in lower or "space" in lower or "left" in lower or "free" in lower):
            return "SYSTEM_QUERY", 1.0

        # Pre-classifier rule: file info / size / metadata → force FILE_OP
        _FILE_PATTERNS = (
            "how big is ", "file size", "size of file", "size of the file",
            "how large is ", "how much space", "when was file",
            "when was the file", "file info", "file details",
            "how many files", "disk usage",
        )
        if any(p in lower for p in _FILE_PATTERNS):
            return "FILE_OP", 1.0

        # Fast keyword override — startswith check for command prefixes
        for intent, prefixes in self._START_PREFIXES.items():
            if any(lower.startswith(kw) for kw in prefixes):
                return intent, 1.0

        # Phrase check — can appear anywhere in the sentence
        for intent, phrases in self._ANYWHERE_PHRASES.items():
            if any(phrase in lower for phrase in phrases):
                return intent, 1.0

        query_emb = self.model.encode([lower], normalize_embeddings=True, show_progress_bar=False)
        # Cosine similarity (embeddings are normalised, so dot product = cosine)
        similarities = np.dot(self._all_embeddings, query_emb.T).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_intent = self._intent_embeddings[best_idx][0]

        if best_score < self.confidence_threshold:
            return "CHAT", best_score

        return best_intent, best_score
