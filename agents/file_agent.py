"""File Agent — Complete file system access, search, read, write, watch, and analysis."""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import PyPDF2
from docx import Document
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _ChangeHandler(FileSystemEventHandler):
    """Internal watchdog event handler."""

    def __init__(self, callback) -> None:
        self.callback = callback

    def on_any_event(self, event: FileSystemEvent) -> None:
        if self.callback and not event.is_directory:
            self.callback(event)


class FileAgent:
    """Complete file system access: read, search, write, watch, and analyze files.

    Can handle the entire 1TB SSD.
    """

    READABLE_EXTENSIONS: set[str] = {
        ".txt", ".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml",
        ".html", ".css", ".csv", ".xml", ".ini", ".cfg", ".toml",
        ".java", ".cpp", ".c", ".h", ".rs", ".go", ".sh", ".bat",
        ".pdf", ".docx", ".log", ".env", ".gitignore", ".dockerfile",
    }

    SKIP_DIRS: set[str] = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "$Recycle.Bin", "Windows", "System Volume Information",
        "ProgramData", "Recovery",
    }

    COMMAND_SYNONYMS: tuple[tuple[str, str], ...] = (
        ("folder name ", "folder named "),
        ("directory name ", "directory named "),
        ("make dir", "make directory"),
        ("create dir", "create directory"),
        ("put it in", "move it to"),
        ("put it on", "move it to"),
        ("put this in", "move this to"),
        ("put this on", "move this to"),
        ("desktop folder", "desktop"),
        ("downloads folder", "downloads"),
        ("documents folder", "documents"),
    )

    def __init__(self, config: dict, on_change_callback=None) -> None:
        """Initialize the file agent.

        Args:
            config: Full JARVIS configuration dictionary.
            on_change_callback: Called on file system changes in watched dirs.
        """
        self.root_paths = [Path(p) for p in config["agent"]["root_paths"]]
        self.watch_dirs: list[str] = config["agent"]["watch_dirs"]
        self.max_chars: int = config["agent"]["max_file_read_chars"]
        self.max_results: int = config["agent"]["max_search_results"]
        self._observer: Observer | None = None
        self._on_change = on_change_callback

    @classmethod
    def _normalize_command(cls, text: str) -> str:
        """Normalize common file-operation phrasing."""
        lower = text.lower().strip()
        for src, dst in cls.COMMAND_SYNONYMS:
            lower = lower.replace(src, dst)
        return " ".join(lower.split())

    # ── SEARCH ────────────────────────────────────────────────────────────────

    def _search_files(
        self,
        query: str = "",
        root: str = "C:/Users",
        extensions: list[str] | None = None,
    ) -> list[Path]:
        """Search files by optional name substring and/or extension list.

        Args:
            query: Optional substring to match in filenames (empty = any).
            root: Root directory to search from.
            extensions: Optional list of extensions to filter by.

        Returns:
            List of matching Paths (up to max_results).
        """
        results: list[Path] = []
        root_path = Path(root)
        pattern = f"*{query}*" if query else "*"
        try:
            for p in root_path.rglob(pattern):
                if any(skip in p.parts for skip in self.SKIP_DIRS):
                    continue
                if not p.is_file():
                    continue
                if extensions and p.suffix.lower() not in extensions:
                    continue
                results.append(p)
                if len(results) >= self.max_results:
                    break
        except (PermissionError, OSError):
            pass
        except Exception as exc:
            logger.error("_search_files error: %s", exc)
        return results

    def search_by_name(
        self,
        query: str,
        root: str = "C:/",
        extensions: list[str] | None = None,
    ) -> list[Path]:
        """Recursive glob search for files by name.

        Args:
            query: Substring to match in filenames.
            root: Root directory to search from.
            extensions: Optional list of extensions to filter.

        Returns:
            List of matching file paths (up to max_results).
        """
        results: list[Path] = []
        root_path = Path(root)
        try:
            for p in root_path.rglob(f"*{query}*"):
                if any(skip in p.parts for skip in self.SKIP_DIRS):
                    continue
                if p.is_file():
                    if extensions and p.suffix.lower() not in extensions:
                        continue
                    results.append(p)
                    if len(results) >= self.max_results:
                        break
        except PermissionError:
            pass
        except Exception as exc:
            logger.error("search_by_name error: %s", exc)
        return results

    def search_by_content(
        self,
        query: str,
        root: str = "C:/Users",
        extensions: list[str] | None = None,
    ) -> list[dict]:
        """Search inside readable files for a query string.

        Args:
            query: Text to search for inside files.
            root: Root directory to search from.
            extensions: Optional list of extensions to filter.

        Returns:
            List of dicts: path, line_number, line_content, context.
        """
        results: list[dict] = []
        root_path = Path(root)
        allowed = extensions or self.READABLE_EXTENSIONS

        try:
            for p in root_path.rglob("*"):
                if any(skip in p.parts for skip in self.SKIP_DIRS):
                    continue
                if not p.is_file() or p.suffix.lower() not in allowed:
                    continue
                if p.stat().st_size > 10 * 1024 * 1024:  # skip > 10MB
                    continue
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if query.lower() in line.lower():
                            results.append({
                                "path": str(p),
                                "line_number": i,
                                "line_content": line.strip()[:200],
                                "context": "",
                            })
                            if len(results) >= self.max_results:
                                return results
                except Exception:
                    continue
        except PermissionError:
            pass
        except Exception as exc:
            logger.error("search_by_content error: %s", exc)
        return results

    # ── READ ──────────────────────────────────────────────────────────────────

    def read_file(self, path: str) -> str:
        """Read a file's contents, detecting type automatically.

        Args:
            path: File path string.

        Returns:
            File content text (truncated to max_chars if needed).

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If access is denied.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            content = self._read_pdf(p)
        elif suffix == ".docx":
            content = self._read_docx(p)
        else:
            content = p.read_text(encoding="utf-8", errors="ignore")

        if len(content) > self.max_chars:
            content = content[: self.max_chars] + "\n[TRUNCATED]"
        return content

    def _read_pdf(self, path: Path) -> str:
        """Extract text from a PDF file.

        Args:
            path: Path to the PDF.

        Returns:
            Extracted text joined by newlines.
        """
        reader = PyPDF2.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _read_docx(self, path: Path) -> str:
        """Extract text from a DOCX file.

        Args:
            path: Path to the DOCX.

        Returns:
            All paragraph text joined by newlines.
        """
        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)

    # ── WRITE & ORGANIZE ──────────────────────────────────────────────────────

    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file, creating parent dirs if needed.

        Args:
            path: Target file path.
            content: Content to write.

        Returns:
            True on success.
        """
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return True
        except Exception as exc:
            logger.error("write_file error: %s", exc)
            return False

    def move_file(self, src: str, dst: str) -> bool:
        """Move a file from src to dst.

        Args:
            src: Source file path.
            dst: Destination file path.

        Returns:
            True on success.
        """
        try:
            src_p = Path(src)
            dst_p = Path(dst)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            src_p.rename(dst_p)
            return True
        except Exception as exc:
            logger.error("move_file error: %s", exc)
            return False

    def delete_file(self, path: str) -> bool:
        """Move a file to the recycle bin (safe delete).

        Args:
            path: File path to delete.

        Returns:
            True on success.
        """
        try:
            from send2trash import send2trash
            send2trash(path)
            return True
        except Exception as exc:
            logger.error("delete_file error: %s", exc)
            return False

    def create_directory(self, path: str) -> bool:
        """Create a directory (and parents).

        Args:
            path: Directory path.

        Returns:
            True on success.
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as exc:
            logger.error("create_directory error: %s", exc)
            return False

    # ── ANALYZE ───────────────────────────────────────────────────────────────

    def get_directory_summary(self, path: str) -> dict:
        """Summarize a directory's contents.

        Args:
            path: Directory path.

        Returns:
            Dict with total_files, total_size_mb, file_types, largest_files,
            most_recent, subdirectories.
        """
        root = Path(path)
        if not root.is_dir():
            return {"error": f"Not a directory: {path}"}

        total_files = 0
        total_size = 0
        file_types: dict[str, int] = defaultdict(int)
        files_by_size: list[tuple[Path, int]] = []
        files_by_time: list[tuple[Path, float]] = []
        subdirs: list[str] = []

        for item in root.iterdir():
            if item.is_dir():
                subdirs.append(item.name)
            elif item.is_file():
                total_files += 1
                size = item.stat().st_size
                total_size += size
                file_types[item.suffix.lower() or "(no ext)"] += 1
                files_by_size.append((item, size))
                files_by_time.append((item, item.stat().st_mtime))

        files_by_size.sort(key=lambda x: x[1], reverse=True)
        files_by_time.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "file_types": dict(file_types),
            "largest_files": [
                {"name": f.name, "size_mb": round(s / 1024 / 1024, 2)}
                for f, s in files_by_size[:5]
            ],
            "most_recent": [
                {"name": f.name, "modified": f.stat().st_mtime}
                for f, _ in files_by_time[:5]
            ],
            "subdirectories": subdirs,
        }

    def find_duplicates(self, root: str) -> list[list[str]]:
        """Find duplicate files by MD5 hash.

        Args:
            root: Root directory to scan.

        Returns:
            List of groups (lists) of duplicate file paths.
        """
        hashes: dict[str, list[str]] = defaultdict(list)
        root_path = Path(root)

        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            if p.stat().st_size > 100 * 1024 * 1024:  # skip > 100MB
                continue
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()
                hashes[h].append(str(p))
            except Exception:
                continue

        return [paths for paths in hashes.values() if len(paths) > 1]

    # ── WATCH ─────────────────────────────────────────────────────────────────

    def start_watching(self) -> None:
        """Start watchdog Observer on all configured watch directories."""
        if self._observer is not None:
            return
        self._observer = Observer()
        handler = _ChangeHandler(self._on_change)
        for d in self.watch_dirs:
            if Path(d).exists():
                self._observer.schedule(handler, d, recursive=True)
                logger.info("Watching directory: %s", d)
        self._observer.start()

    def stop_watching(self) -> None:
        """Stop the watchdog observer."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    # ── NATURAL LANGUAGE HANDLER ──────────────────────────────────────────────

    # Maps informal words to image/video/doc file extensions
    _TYPE_EXTENSIONS: dict[str, list[str]] = {
        "pic":        [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tiff"],
        "photo":      [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tiff"],
        "image":      [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".heic"],
        "screenshot": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
        "selfie":     [".jpg", ".jpeg", ".png", ".heic"],
        "video":      [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"],
        "movie":      [".mp4", ".mkv", ".avi", ".mov", ".wmv"],
        "music":      [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
        "song":       [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
        "document":   [".pdf", ".docx", ".doc", ".txt", ".odt"],
        "pdf":        [".pdf"],
        "word":       [".docx", ".doc"],
        "excel":      [".xlsx", ".xls", ".csv"],
        "zip":        [".zip", ".rar", ".7z", ".tar", ".gz"],
    }

    # Maps drive keywords to root paths
    _DRIVE_MAP: dict[str, str] = {
        "e drive": "E:/", "e:": "E:/", "on e": "E:/",
        "d drive": "D:/", "d:": "D:/", "on d": "D:/",
        "f drive": "F:/", "f:": "F:/", "on f": "F:/",
        "c drive": "C:/", "c:": "C:/", "on c": "C:/",
        "desktop": str(Path.home() / "Desktop"),
        "downloads": str(Path.home() / "Downloads"),
        "documents": str(Path.home() / "Documents"),
        "pictures": str(Path.home() / "Pictures"),
        "music": str(Path.home() / "Music"),
        "videos": str(Path.home() / "Videos"),
    }

    # Folder aliases for "list files in <folder>" style commands
    _FOLDER_ALIASES: dict[str, Path] = {
        "downloads": Path.home() / "Downloads",
        "download": Path.home() / "Downloads",
        "desktop": Path.home() / "Desktop",
        "documents": Path.home() / "Documents",
        "pictures": Path.home() / "Pictures",
        "music": Path.home() / "Music",
        "videos": Path.home() / "Videos",
        "home": Path.home(),
    }

    def _detect_root(self, text: str) -> str:
        """Detect which drive/folder the user is referring to."""
        low = text.lower()
        for key, path in self._DRIVE_MAP.items():
            if key in low:
                return path
        return "C:/Users"  # default to user dirs

    def _detect_file_type(self, text: str) -> list[str] | None:
        """Map informal type words to a list of extensions."""
        low = text.lower()
        for keyword, exts in self._TYPE_EXTENSIONS.items():
            if keyword in low:
                return exts
        return None

    async def handle(self, user_input: str, context: dict) -> str:
        """Parse and execute a natural language file command.

        Args:
            user_input: The user's natural language request.
            context: Context dict from ContextBuilder.

        Returns:
            Response string.
        """
        import re
        nl = self._normalize_command(user_input)

        # ── "list files in <folder>" / "show files in downloads" ──────────
        list_match = re.search(
            r'(?:list|show|what(?:\'s| is) in|contents? of)\s+'
            r'(?:the\s+)?(?:files?\s+(?:in|on|of)\s+)?(?:the\s+)?(\w+)',
            nl,
        )
        if list_match and any(w in nl for w in ("list", "show", "what's in", "what is in", "contents")):
            folder_name = list_match.group(1).strip()
            folder_path = self._FOLDER_ALIASES.get(folder_name)
            if folder_path is None:
                # Try _DRIVE_MAP as fallback
                for key, path_str in self._DRIVE_MAP.items():
                    if folder_name in key:
                        folder_path = Path(path_str)
                        break
            if folder_path and folder_path.is_dir():
                items = sorted(folder_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                files = [p for p in items if p.is_file()][:30]
                dirs = [p for p in items if p.is_dir()][:10]
                lines: list[str] = []
                if dirs:
                    lines.append(f"Folders ({len(dirs)}):")
                    lines.extend(f"  📁 {d.name}" for d in dirs)
                if files:
                    lines.append(f"Files ({len(files)}):")
                    for f in files:
                        size = f.stat().st_size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size / 1024 / 1024:.1f} MB"
                        lines.append(f"  📄 {f.name}  ({size_str})")
                if not files and not dirs:
                    return f"{folder_path} is empty."
                return f"Contents of {folder_path}:\n" + "\n".join(lines)
            if folder_path:
                return f"Folder not found: {folder_path}"

        # ── "create a folder named X on desktop" ───────────────────────────
        folder_cmd = re.match(
            r'(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory)\b(.+)?$',
            nl,
        )
        if folder_cmd:
            remainder = (folder_cmd.group(1) or "").strip()
            location_key = ""

            followup_loc = re.search(
                r'(?:^|\s+)and\s+(?:move|put)\s+it\s+to\s+(desktop|downloads|documents)\s*$',
                remainder,
            )
            if followup_loc:
                location_key = followup_loc.group(1)
                remainder = remainder[:followup_loc.start()].strip()

            direct_loc = re.search(
                r'(?:^|\s+)(?:in|on|at|to|inside|under)\s+(desktop|downloads|documents)\s*$',
                remainder,
            )
            if direct_loc:
                location_key = direct_loc.group(1)
                remainder = remainder[:direct_loc.start()].strip()

            remainder = re.sub(r'^(?:named|name)\s+', '', remainder).strip()
            folder_name = remainder.strip(' ."') or "New Folder"
            base_path = self._FOLDER_ALIASES.get(location_key) if location_key else None
            if base_path is None:
                base_path = Path.cwd()
            target = base_path / folder_name
            if self.create_directory(str(target)):
                return f"Created folder: {target}"
            return f"Failed to create folder: {target}"

        if any(w in nl for w in ("find", "search", "where", "locate", "show me")):
            # Detect drive / folder
            root = self._detect_root(nl)

            # Detect file type (pic, video, doc, etc.)
            extensions = self._detect_file_type(nl)

            # Extract filename/query — handles "where my X", "where is my X", "find X"
            match = re.search(
                r"(?:find|search for|where(?:\s+is)?|locate|show me)\s+"
                r"(?:my\s+|the\s+|all\s+)?(.+?)(?:\s+(?:file|files|on|in|at)\b|\s*$)",
                nl,
            )
            query = match.group(1).strip() if match else ""

            # If query is a generic type word, search by extension only (not name)
            if query in self._TYPE_EXTENSIONS or not query:
                query = ""  # search all by extension

            if "content" in nl or "inside" in nl or "containing" in nl:
                ext_match = re.search(r"\.(\w+)\s+files?", nl)
                filter_exts = [f".{ext_match.group(1)}"] if ext_match else extensions
                results = self.search_by_content(query or ".", root=root, extensions=filter_exts)
                if results:
                    lines = [f"  {r['path']}:{r['line_number']}: {r['line_content']}" for r in results[:10]]
                    return f"Found {len(results)} matches:\n" + "\n".join(lines)
                return f"No matches found for '{query}'."

            # Name/extension search
            results = self._search_files(query=query, root=root, extensions=extensions)
            if results:
                lines = [f"  {p}" for p in results[:20]]
                type_label = f" ({', '.join(extensions[:3])})" if extensions else ""
                return (
                    f"Found {len(results)} file(s){type_label} in {root}:\n"
                    + "\n".join(lines)
                )
            type_label = f" {query}" if query else (f" {extensions[0]}" if extensions else "")
            return f"No{type_label} files found in {root}."

        if "read" in nl:
            import re
            match = re.search(r"read\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s*$)", nl)
            if match:
                target = match.group(1).strip()
                # Strip leading "file" / "file called" if present
                target = re.sub(r'^(?:file\s+(?:called\s+)?)', '', target).strip()
                # Try local data/ directory first, then full search
                local_path = Path("data") / target
                if local_path.exists():
                    try:
                        content = self.read_file(str(local_path))
                        return f"Contents of {local_path}:\n\n{content}"
                    except Exception as exc:
                        return f"Could not read file: {exc}"
                # Fallback: search by name
                results = self.search_by_name(target)
                if results:
                    try:
                        content = self.read_file(str(results[0]))
                        return f"Contents of {results[0]}:\n\n{content}"
                    except Exception as exc:
                        return f"Could not read file: {exc}"
                return f"Could not find file: {target}"

        if "summarize" in nl or "summary" in nl:
            import re
            match = re.search(r"(?:summarize|summary of)\s+(?:my\s+)?(.+)", nl)
            if match:
                target = match.group(1).strip()
                path = Path(target)
                if not path.is_dir():
                    # Try common paths
                    user_home = Path.home()
                    for candidate in [user_home / target, user_home / target.title()]:
                        if candidate.is_dir():
                            path = candidate
                            break
                if path.is_dir():
                    summary = self.get_directory_summary(str(path))
                    return (
                        f"Directory: {path}\n"
                        f"Total files: {summary['total_files']}\n"
                        f"Total size: {summary['total_size_mb']} MB\n"
                        f"File types: {summary['file_types']}\n"
                        f"Largest files: {summary['largest_files']}\n"
                        f"Subdirectories: {summary['subdirectories']}"
                    )
                return f"Directory not found: {target}"

        if "duplicate" in nl:
            import re
            match = re.search(r"(?:duplicate|duplicates)\s+(?:in|on|files)\s+(?:my\s+)?(.+)", nl)
            target = match.group(1).strip() if match else str(Path.home() / "Desktop")
            dupes = self.find_duplicates(target)
            if dupes:
                lines = []
                for group in dupes[:5]:
                    lines.append("  Duplicates:\n" + "\n".join(f"    - {p}" for p in group))
                return f"Found {len(dupes)} groups of duplicate files:\n" + "\n".join(lines)
            return "No duplicate files found."

        # Save / create / write file
        if any(w in nl for w in ("save", "write", "create", "store")):
            # Extract filename
            fname_match = re.search(
                r'(?:to file|to disk|as|create file|create)\s+([\w.\-/\\]+\.\w+)', nl
            )
            if not fname_match:
                fname_match = re.search(r'([\w.\-]+\.(?:txt|csv|json|md|html|xlsx|pdf|docx))', nl)
            if fname_match:
                filename = fname_match.group(1)
                filepath = Path("data") / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                # Extract content from user input ("with content X" / "content X" / "saying X")
                content = ""
                content_match = re.search(
                    r'(?:with content|content|saying|containing|text)\s+(.+?)$',
                    user_input, re.I
                )
                if content_match:
                    content = content_match.group(1).strip()
                if not content:
                    content = context.get("content", "")
                if not content:
                    content = f"File created by JARVIS for: {user_input}"
                filepath.write_text(content, encoding="utf-8")
                return f"Created {filepath} with content: {content[:100]}"
            return f"Could not determine filename from: '{user_input}'. Try: 'save to file report.txt'"

        # ── File info / size queries ──────────────────────────────────────
        file_info_match = re.search(
            r'(?:how (?:big|large) is (?:the )?(?:file )?|file size (?:of )?|'
            r'size of (?:the )?(?:file )?|file info (?:for )?|file details (?:for )?)'
            r'(.+?)(?:\s*\??\s*)$',
            nl,
        )
        if file_info_match:
            target = file_info_match.group(1).strip()
            # Strip leading "file" if present
            target = re.sub(r'^(?:file\s+(?:called\s+)?)', '', target).strip()
            # Try local data/ directory first (fast path)
            local_path = Path("data") / target
            if local_path.exists():
                results = [local_path]
            else:
                results = self.search_by_name(target, root=str(Path.cwd()))
            if results:
                p = results[0]
                stat = p.stat()
                size = stat.st_size
                if size < 1024:
                    size_str = f"{size} bytes"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                elif size < 1024 * 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                else:
                    size_str = f"{size / 1024 / 1024 / 1024:.2f} GB"
                from datetime import datetime
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                created = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M")
                return (
                    f"File: {p}\n"
                    f"Size: {size_str}\n"
                    f"Type: {p.suffix or '(no extension)'}\n"
                    f"Modified: {modified}\n"
                    f"Created: {created}"
                )
            return f"File not found: {target}"

        return f"I understood this as a file operation but couldn't determine the specific action. You said: '{user_input}'"
