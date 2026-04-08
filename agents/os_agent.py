"""OS Agent — Full Windows 11 OS control with autonomous GUI automation."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path

import GPUtil
import psutil
import pyautogui
import pygetwindow as gw
import pyperclip

logger = logging.getLogger(__name__)

# Prevent pyautogui from raising exception on screen edge
pyautogui.FAILSAFE = True


def _discover_start_apps() -> dict[str, str]:
    """Query Windows Start Menu for all installed apps and their AppIDs.

    Returns:
        Dict mapping lowercase app display name → AppID string.
    """
    apps: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-StartApps | ForEach-Object { $_.Name + '|||' + $_.AppID }"],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.strip().splitlines():
            if "|||" in line:
                name, app_id = line.split("|||", 1)
                apps[name.strip().lower()] = app_id.strip()
    except Exception as exc:
        logger.warning("Could not enumerate Start Menu apps: %s", exc)
    return apps


class OSAgent:
    """Controls the entire Windows 11 OS: processes, windows, files, GUI.

    The most powerful agent — it can do anything the user can do on their PC.
    """

    # Friendly aliases → canonical name used to look up in discovered Start Menu apps
    APP_ALIASES: dict[str, list[str]] = {
        "chrome": ["google chrome"],
        "google chrome": ["google chrome"],
        "firefox": ["firefox", "mozilla firefox"],
        "edge": ["microsoft edge"],
        "microsoft edge": ["microsoft edge"],
        "vscode": ["visual studio code"],
        "vs code": ["visual studio code"],
        "visual studio code": ["visual studio code"],
        "notepad": ["notepad"],
        "terminal": ["terminal", "windows terminal"],
        "windows terminal": ["terminal", "windows terminal"],
        "powershell": ["powershell", "windows powershell"],
        "cmd": ["command prompt"],
        "command prompt": ["command prompt"],
        "explorer": ["file explorer"],
        "file explorer": ["file explorer"],
        "task manager": ["task manager"],
        "discord": ["discord"],
        "spotify": ["spotify"],
        "telegram": ["telegram", "telegram desktop"],
        "slack": ["slack"],
        "zoom": ["zoom", "zoom workplace"],
        "obs": ["obs studio"],
        "vlc": ["vlc", "vlc media player"],
        "word": ["word"],
        "excel": ["excel"],
        "powerpoint": ["powerpoint"],
        "paint": ["paint"],
        "calculator": ["calculator"],
        "brave": ["brave"],
        "steam": ["steam"],
        "whatsapp": ["whatsapp"],
        "settings": ["settings"],
        "teams": ["microsoft teams", "teams"],
        "microsoft teams": ["microsoft teams", "teams"],
        "photos": ["photos"],
        "mail": ["mail", "mail and calendar"],
        "store": ["microsoft store"],
        "clock": ["clock", "alarms & clock"],
        "camera": ["camera"],
        "maps": ["maps"],
        "weather": ["weather"],
        "news": ["news"],
    }

    COMMAND_SYNONYMS: tuple[tuple[str, str], ...] = (
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
        ("open browser", "open chrome"),
        ("start browser", "open chrome"),
        ("take screenshot", "take a screenshot"),
        ("capture screen", "take a screenshot"),
        ("screen shot", "screenshot"),
        ("turn up volume", "increase volume"),
        ("turn down volume", "decrease volume"),
        ("raise volume", "increase volume"),
        ("lower volume", "decrease volume"),
        ("sound off", "mute"),
        ("sound on", "unmute"),
        ("play next song", "next song"),
        ("skip to next song", "next song"),
        ("go to next song", "next song"),
        ("go to previous song", "previous song"),
        ("play previous song", "previous song"),
        ("un pause", "resume"),
        ("continue music", "resume music"),
        ("stop the song", "pause music"),
    )

    def __init__(self, config: dict) -> None:
        """Initialize the OS agent.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.config = config
        self.screenshot_dir = Path(config.get("vision", {}).get("screenshot_path", "data/screenshots"))
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        # Discover all installed apps from Start Menu at startup
        self._start_apps = _discover_start_apps()
        logger.info("Discovered %d Start Menu apps.", len(self._start_apps))
        self.vision = None  # Will be set by main.py after ImageAnalyzer is created

    # ── PROCESS MANAGEMENT ────────────────────────────────────────────────────

    def list_processes(self) -> list[dict]:
        """Return running processes sorted by CPU usage descending.

        Returns:
            List of dicts: name, pid, cpu_percent, memory_mb.
        """
        procs: list[dict] = []
        for proc in psutil.process_iter(["name", "pid", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                procs.append({
                    "name": info["name"],
                    "pid": info["pid"],
                    "cpu_percent": info["cpu_percent"] or 0.0,
                    "memory_mb": round((info["memory_info"].rss / 1024 / 1024), 1) if info["memory_info"] else 0,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
        return procs

    def kill_process(self, name_or_pid: str | int) -> int:
        """Kill a process by name (all matching) or PID.

        Args:
            name_or_pid: Process name string or integer PID.

        Returns:
            Number of processes killed (0 means none found/killed).
        """
        try:
            pid = int(name_or_pid)
            proc = psutil.Process(pid)
            proc.terminate()
            logger.info("Killed process PID %d", pid)
            return 1
        except (ValueError, TypeError):
            pass
        except psutil.NoSuchProcess:
            return 0

        name = str(name_or_pid).lower()
        count = 0
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if proc.info["name"] and name in proc.info["name"].lower():
                    proc.terminate()
                    count += 1
                    logger.info("Killed process: %s (PID %d)", proc.info["name"], proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return count

    def start_process(self, app_name: str) -> bool:
        """Launch an application by name.

        Uses a multi-step strategy:
          1. Look up in Start Menu apps discovered at startup
          2. PowerShell Start-Process (searches PATH, App Paths, Start Menu)
          3. os.startfile fallback

        Args:
            app_name: Application name (e.g., "chrome", "whatsapp", "discord").

        Returns:
            True if launched successfully.
        """
        key = app_name.lower().strip()

        # Handle ms-settings: and other URI schemes
        if ":" in key and not key.endswith(".exe"):
            try:
                subprocess.Popen(
                    ["explorer.exe", app_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Launched URI: %s", app_name)
                return True
            except Exception as exc:
                logger.error("Failed to launch URI %s: %s", app_name, exc)

        # 1. Try Start Menu app discovery (most reliable for ALL apps)
        app_id = self._find_start_app(key)
        if app_id:
            return self._launch_by_app_id(app_id, key)

        # 2. PowerShell Start-Process (searches PATH + App Paths + system)
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Start-Process '{app_name}'"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info("Started '%s' via PowerShell Start-Process.", app_name)
                return True
            logger.debug("Start-Process failed for '%s': %s", app_name, result.stderr[:200])
        except Exception as exc:
            logger.debug("Start-Process exception for '%s': %s", app_name, exc)

        # 3. os.startfile fallback
        try:
            os.startfile(app_name)
            logger.info("Started '%s' via os.startfile.", app_name)
            return True
        except Exception:
            pass

        return False

    def _find_start_app(self, key: str) -> str | None:
        """Find an app's AppID from the Start Menu discovery cache.

        Args:
            key: Lowercase app name or alias.

        Returns:
            AppID string or None.
        """
        # Direct match in Start Menu
        if key in self._start_apps:
            return self._start_apps[key]

        # Check aliases
        aliases = self.APP_ALIASES.get(key, [])
        for alias in aliases:
            if alias in self._start_apps:
                return self._start_apps[alias]

        # Fuzzy: check if key is a substring of any Start Menu entry
        for name, app_id in self._start_apps.items():
            if key in name:
                return app_id

        return None

    def _launch_by_app_id(self, app_id: str, display_name: str) -> bool:
        """Launch an app by its Start Menu AppID.

        Args:
            app_id: The AppID from Get-StartApps.
            display_name: Human-readable name for logging.

        Returns:
            True if launched successfully.
        """
        shell_uri = f"shell:AppsFolder\\{app_id}"
        try:
            subprocess.Popen(
                ["explorer.exe", shell_uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Launched '%s' via AppID: %s", display_name, app_id)
            return True
        except Exception as exc:
            logger.error("Failed to launch '%s' via AppID: %s", display_name, exc)
            return False

    def _launch_store_app(self, app_family_id: str, display_name: str) -> bool:
        """Launch a Windows Store / UWP application.

        Args:
            app_family_id: The PackageFamilyName!AppId string.
            display_name: Human-readable app name for logging.

        Returns:
            True if launched successfully.
        """
        shell_uri = f"shell:AppsFolder\\{app_family_id}"
        try:
            subprocess.Popen(
                ["explorer.exe", shell_uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Launched Store app '%s': %s", display_name, shell_uri)
            return True
        except Exception as exc:
            logger.error("Failed to launch Store app '%s': %s", display_name, exc)
            return False

    # ── WINDOW MANAGEMENT ─────────────────────────────────────────────────────

    def get_active_window(self) -> str:
        """Return the title of the currently active window.

        Returns:
            Window title string, or empty string on failure.
        """
        try:
            win = gw.getActiveWindow()
            return win.title if win else ""
        except Exception:
            return ""

    def list_windows(self) -> list[str]:
        """Return titles of all visible windows.

        Returns:
            List of window title strings.
        """
        try:
            return [w.title for w in gw.getAllWindows() if w.title.strip()]
        except Exception:
            return []

    def focus_window(self, title_contains: str) -> bool:
        """Bring a window containing the given text to the foreground.

        Args:
            title_contains: Substring to match in window titles.

        Returns:
            True if a matching window was found and activated.
        """
        try:
            for win in gw.getAllWindows():
                if title_contains.lower() in win.title.lower():
                    win.activate()
                    return True
        except Exception as exc:
            logger.error("focus_window error: %s", exc)
        return False

    def close_window(self, title_contains: str) -> bool:
        """Close a window containing the given text.

        Args:
            title_contains: Substring to match in window titles.

        Returns:
            True if a matching window was found and closed.
        """
        try:
            for win in gw.getAllWindows():
                if title_contains.lower() in win.title.lower():
                    win.close()
                    return True
        except Exception as exc:
            logger.error("close_window error: %s", exc)
        return False

    # ── SYSTEM STATE ──────────────────────────────────────────────────────────

    def get_system_state(self) -> dict:
        """Return a comprehensive snapshot of the system.

        Returns:
            Dict with CPU, RAM, GPU, active window, top processes, disk info.
        """
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\")

        state = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": round(mem.used / 1024 ** 3, 1),
            "ram_total_gb": round(mem.total / 1024 ** 3, 1),
            "ram_percent": mem.percent,
            "gpu_name": "",
            "gpu_vram_used_mb": 0,
            "gpu_vram_total_mb": 0,
            "gpu_load_percent": 0.0,
            "active_window": self.get_active_window(),
            "top_processes": self.list_processes()[:5],
            "disk_free_gb": round(disk.free / 1024 ** 3, 1),
        }

        gpu_info = self.get_gpu_info()
        state.update(gpu_info)
        return state

    def get_gpu_info(self) -> dict:
        """Query NVIDIA GPU info using GPUtil.

        Returns:
            Dict with gpu_name, gpu_vram_used_mb, gpu_vram_total_mb, gpu_load_percent.
        """
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if "nvidia" in gpu.name.lower():
                    return {
                        "gpu_name": gpu.name,
                        "gpu_vram_used_mb": int(gpu.memoryUsed),
                        "gpu_vram_total_mb": int(gpu.memoryTotal),
                        "gpu_load_percent": round(gpu.load * 100, 1),
                    }
        except Exception as exc:
            logger.debug("GPU info unavailable: %s", exc)
        return {
            "gpu_name": "Unknown",
            "gpu_vram_used_mb": 0,
            "gpu_vram_total_mb": 0,
            "gpu_load_percent": 0.0,
        }

    # ── GUI AUTOMATION ────────────────────────────────────────────────────────

    def take_screenshot(self, save_path: str | None = None) -> str:
        """Capture the screen and save to disk.

        Args:
            save_path: Optional file path. Auto-generated if None.

        Returns:
            The file path of the saved screenshot.
        """
        import datetime
        if save_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.screenshot_dir / f"screenshot_{ts}.png")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img = pyautogui.screenshot()
        img.save(save_path)
        logger.info("Screenshot saved: %s", save_path)
        return save_path

    def type_text(self, text: str) -> None:
        """Type text using keyboard simulation.

        Args:
            text: The text to type.
        """
        pyautogui.typewrite(text, interval=0.05)

    def press_keys(self, *keys: str) -> None:
        """Press a keyboard hotkey combination.

        Args:
            keys: Key names (e.g., "ctrl", "c").
        """
        pyautogui.hotkey(*keys)

    # ── AUTONOMOUS GUI AUTOMATION ─────────────────────────────────────────────

    def _wait_for_window(self, title_substr: str, timeout: int = 15):
        """Wait for a window whose title contains the given substring.

        Args:
            title_substr: Case-insensitive substring to match.
            timeout: Max seconds to wait.

        Returns:
            Matching pygetwindow Window object, or None on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            for win in gw.getAllWindows():
                if title_substr.lower() in win.title.lower() and win.title.strip():
                    return win
            time.sleep(0.5)
        return None

    def _activate_window_obj(self, win) -> bool:
        """Bring a pygetwindow Window object to the foreground.

        Args:
            win: A pygetwindow Window object.

        Returns:
            True if the window was activated successfully.
        """
        try:
            if win.isMinimized:
                win.restore()
                time.sleep(0.5)
            win.activate()
            time.sleep(0.5)
            return True
        except Exception as exc:
            logger.warning("Could not activate window: %s", exc)
            return False

    def _safe_type(self, text: str) -> None:
        """Type text reliably via clipboard paste (handles Unicode).

        Args:
            text: The text string to type.
        """
        old_clip = ""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            pass
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.3)
        try:
            pyperclip.copy(old_clip)
        except Exception:
            pass

    # ── COMPOUND COMMAND PARSING ──────────────────────────────────────────────

    MESSAGING_APPS = {"whatsapp", "telegram"}

    def _parse_messaging_command(self, text: str) -> dict | None:
        """Extract app, contact, and message from a compound messaging command.

        Supported patterns:
          - "open whatsapp and send hi to rifana"
          - "send hi to rifana on whatsapp"
          - "open whatsapp and send a message to rifana saying hello"
          - "message rifana hello on telegram"
          - "tell rifana hi on whatsapp"
          - "open telegram and message john hello"

        Returns:
            Dict with 'app', 'contact', 'message' keys, or None.
        """
        nl = text.lower().strip()

        # Must mention a known messaging app
        app = None
        for name in self.MESSAGING_APPS:
            if name in nl:
                app = name
                break
        if not app:
            return None

        # Must contain a messaging verb (not just "open whatsapp")
        has_verb = any(v in nl for v in (
            "send ", "message ", "msg ", "dm ", "tell ", "text ", "write ",
        ))
        if not has_verb:
            return None

        # Strip the app reference and connection words to isolate the action
        cleaned = nl
        cleaned = re.sub(
            r'(?:open|launch|start)\s+' + app + r'\s+and\s+', '', cleaned,
        )
        cleaned = re.sub(
            r'\s+(?:on|in|via|using|through)\s+' + app + r'\b', '', cleaned,
        )
        cleaned = re.sub(r'^' + app + r'\s+', '', cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            return None

        # Pattern A: "send [a] message to <contact> saying <message>"
        m = re.match(
            r'(?:send|type)\s+(?:a\s+)?message\s+to\s+(\w+)'
            r'\s+(?:saying|that\s+says|with)\s+(.+?)\s*$',
            cleaned,
        )
        if m:
            return {"app": app, "contact": m.group(1), "message": m.group(2).strip()}

        # Pattern B: "send <message> to <contact>"
        m = re.match(
            r'(?:send|text|type)\s+(.+?)\s+to\s+(\w+)\s*$', cleaned,
        )
        if m:
            msg = m.group(1).strip()
            contact = m.group(2).strip()
            # Strip article "a" from short messages like "send a hi to …"
            if msg.startswith('a ') and len(msg.split()) <= 2:
                msg = re.sub(r'^a\s+', '', msg)
            if msg and msg not in ("message", "a message"):
                return {"app": app, "contact": contact, "message": msg}

        # Pattern C: "message/tell/dm <contact> <message>"
        m = re.match(
            r'(?:message|msg|dm|tell|text|send)\s+(\w+)\s+(.+?)\s*$',
            cleaned,
        )
        if m:
            contact = m.group(1).strip()
            message = m.group(2).strip()
            if contact and message and contact not in ("to", "a"):
                return {"app": app, "contact": contact, "message": message}

        return None

    # ── WHATSAPP AUTOMATION ───────────────────────────────────────────────────

    async def _send_whatsapp_message(self, contact: str, message: str) -> str:
        """Send a message via WhatsApp Desktop using vision-guided GUI automation.

        Flow: open/focus WhatsApp → vision-find search bar → type contact →
              vision-find contact → click it → vision-find message box →
              type message → press Enter.

        Falls back to coordinate-based automation if vision is unavailable.

        Args:
            contact: Display name of the contact.
            message: Message text to send.

        Returns:
            Status string.
        """
        # 1. Open or find WhatsApp
        win = self._wait_for_window("whatsapp", timeout=2)
        if not win:
            self.start_process("whatsapp")
            win = self._wait_for_window("whatsapp", timeout=15)
        if not win:
            return "Could not open WhatsApp. Make sure it is installed."

        # 2. Bring to foreground
        if not self._activate_window_obj(win):
            return "Could not bring WhatsApp to the foreground."
        time.sleep(3)

        # 3. Dismiss any overlay / popup
        pyautogui.press('escape')
        time.sleep(0.5)

        # ── VISION-GUIDED PATH ────────────────────────────────────────────
        if self.vision:
            try:
                return await self._send_whatsapp_vision(contact, message, win)
            except Exception as exc:
                logger.warning("Vision-guided WhatsApp failed: %s — falling back to coordinates", exc)

        # ── COORDINATE FALLBACK ───────────────────────────────────────────
        return await self._send_whatsapp_coords(contact, message, win)

    async def _send_whatsapp_vision(self, contact: str, message: str, win) -> str:
        """Vision-guided WhatsApp send using moondream to find UI elements."""
        # 4. Find and click the search bar using vision
        ok, status = await self.vision.vision_click("WhatsApp search bar or search text box at the top")
        if not ok:
            raise RuntimeError(f"Could not find search bar: {status}")
        time.sleep(0.5)

        # 5. Clear and type the contact name
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        self._safe_type(contact)
        time.sleep(2)  # wait for search results

        # 6. Find and click the contact in search results
        ok, status = await self.vision.vision_click(
            f"the chat or contact named '{contact}' in the search results list"
        )
        if not ok:
            # Fallback: just press down+enter
            pyautogui.press('down')
            time.sleep(0.3)
            pyautogui.press('enter')
        time.sleep(1)

        # 7. Find and click the message input box using vision
        ok, status = await self.vision.vision_click(
            "the message input box or text field at the bottom of the chat where you type messages"
        )
        if not ok:
            raise RuntimeError(f"Could not find message box: {status}")
        time.sleep(0.3)

        # 8. Type the message and send
        self._safe_type(message)
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)

        pyautogui.press('escape')
        logger.info("Vision-sent WhatsApp message to '%s': %s", contact, message)
        return f"Sent '{message}' to {contact} on WhatsApp (vision-guided)."

    async def _send_whatsapp_coords(self, contact: str, message: str, win) -> str:
        """Coordinate-based WhatsApp send (fallback when vision is unavailable)."""
        search_x = win.left + max(210, int(win.width * 0.18))
        search_y = win.top + 80
        pyautogui.click(search_x, search_y)
        time.sleep(0.5)

        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        self._safe_type(contact)
        time.sleep(2)

        pyautogui.press('down')
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(1)

        msg_x = win.left + int(win.width * 0.65)
        msg_y = win.top + win.height - 55
        pyautogui.click(msg_x, msg_y)
        time.sleep(0.3)

        self._safe_type(message)
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)

        pyautogui.press('escape')
        logger.info("Sent WhatsApp message to '%s': %s", contact, message)
        return f"Sent '{message}' to {contact} on WhatsApp."

    # ── TELEGRAM AUTOMATION ───────────────────────────────────────────────────

    async def _send_telegram_message(self, contact: str, message: str) -> str:
        """Send a message via Telegram Desktop using vision-guided GUI automation.

        Falls back to coordinate-based automation if vision is unavailable.
        """
        win = self._wait_for_window("telegram", timeout=2)
        if not win:
            self.start_process("telegram")
            win = self._wait_for_window("telegram", timeout=15)
        if not win:
            return "Could not open Telegram. Make sure it is installed."

        if not self._activate_window_obj(win):
            return "Could not bring Telegram to the foreground."
        time.sleep(3)

        pyautogui.press('escape')
        time.sleep(0.3)

        # ── VISION-GUIDED PATH ────────────────────────────────────────────
        if self.vision:
            try:
                return await self._send_telegram_vision(contact, message, win)
            except Exception as exc:
                logger.warning("Vision-guided Telegram failed: %s — falling back to coordinates", exc)

        # ── COORDINATE FALLBACK ───────────────────────────────────────────
        return await self._send_telegram_coords(contact, message, win)

    async def _send_telegram_vision(self, contact: str, message: str, win) -> str:
        """Vision-guided Telegram send."""
        ok, status = await self.vision.vision_click("Telegram search bar or search icon at the top")
        if not ok:
            raise RuntimeError(f"Could not find search bar: {status}")
        time.sleep(0.5)

        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        self._safe_type(contact)
        time.sleep(2)

        ok, status = await self.vision.vision_click(
            f"the chat or contact named '{contact}' in the search results"
        )
        if not ok:
            pyautogui.press('down')
            time.sleep(0.3)
            pyautogui.press('enter')
        time.sleep(1)

        ok, status = await self.vision.vision_click(
            "the message input box or text field at the bottom of the chat"
        )
        if not ok:
            raise RuntimeError(f"Could not find message box: {status}")
        time.sleep(0.3)

        self._safe_type(message)
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)

        pyautogui.press('escape')
        logger.info("Vision-sent Telegram message to '%s': %s", contact, message)
        return f"Sent '{message}' to {contact} on Telegram (vision-guided)."

    async def _send_telegram_coords(self, contact: str, message: str, win) -> str:
        """Coordinate-based Telegram send (fallback)."""
        search_x = win.left + int(win.width * 0.17)
        search_y = win.top + 55
        pyautogui.click(search_x, search_y)
        time.sleep(0.5)

        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        self._safe_type(contact)
        time.sleep(2)

        pyautogui.press('down')
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(1)

        msg_x = win.left + int(win.width * 0.65)
        msg_y = win.top + win.height - 45
        pyautogui.click(msg_x, msg_y)
        time.sleep(0.3)

        self._safe_type(message)
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)

        pyautogui.press('escape')
        logger.info("Sent Telegram message to '%s': %s", contact, message)
        return f"Sent '{message}' to {contact} on Telegram."

    # ── BROWSER AUTOMATION ────────────────────────────────────────────────────

    async def _open_and_search_browser(self, browser: str, query: str) -> str:
        """Open a browser and search for a query.

        Args:
            browser: Browser name (chrome, brave, edge, firefox).
            query: Search query text.

        Returns:
            Status string.
        """
        win = self._wait_for_window(browser, timeout=2)
        if not win:
            self.start_process(browser)
            win = self._wait_for_window(browser, timeout=10)
        if not win:
            return f"Could not open {browser}."

        if not self._activate_window_obj(win):
            return f"Could not bring {browser} to the foreground."
        time.sleep(2)

        # Focus address bar and search
        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.3)
        self._safe_type(query)
        time.sleep(0.2)
        pyautogui.press('enter')

        logger.info("Searched '%s' in %s", query, browser)
        return f"Searched for '{query}' in {browser}."

    async def _open_and_navigate(self, browser: str, url: str) -> str:
        """Open a browser and navigate to a URL.

        Args:
            browser: Browser name.
            url: Target URL.

        Returns:
            Status string.
        """
        win = self._wait_for_window(browser, timeout=2)
        if not win:
            self.start_process(browser)
            win = self._wait_for_window(browser, timeout=10)
        if not win:
            return f"Could not open {browser}."

        if not self._activate_window_obj(win):
            return f"Could not bring {browser} to the foreground."
        time.sleep(2)

        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.3)
        self._safe_type(url)
        time.sleep(0.2)
        pyautogui.press('enter')

        logger.info("Navigated to '%s' in %s", url, browser)
        return f"Opened {url} in {browser}."

    async def _open_and_type(self, app: str, text: str) -> str:
        """Open an application and type text into it.

        Args:
            app: Application name.
            text: Text to type.

        Returns:
            Status string.
        """
        win = self._wait_for_window(app, timeout=2)
        if not win:
            self.start_process(app)
            win = self._wait_for_window(app, timeout=10)
        if not win:
            return f"Could not open {app}."

        if not self._activate_window_obj(win):
            return f"Could not bring {app} to the foreground."
        time.sleep(1)

        self._safe_type(text)
        logger.info("Typed '%s' into %s", text[:50], app)
        return f"Opened {app} and typed: {text}"

    async def type_in_app(self, app_name: str, text: str) -> str:
        """Open an application, wait for its window, focus it, and type text.

        Steps:
          1. Open the app via start_process (Start Menu / PowerShell / os.startfile).
          2. Wait up to 10s for the window to appear using pygetwindow.
          3. Focus the window (restore if minimised, then activate).
          4. Type the text using clipboard-paste (_safe_type) for Unicode safety.

        Args:
            app_name: Name of the application (e.g. 'notepad', 'wordpad').
            text: The text string to type into the application.

        Returns:
            Confirmation message on success, or error description on failure.
        """
        # 1. Try to find an already-open window first
        win = self._wait_for_window(app_name, timeout=2)
        if not win:
            # Launch the app
            launched = self.start_process(app_name)
            if not launched:
                return f"Could not open {app_name}. Make sure it is installed."
            # Wait for window to appear
            win = self._wait_for_window(app_name, timeout=10)
        if not win:
            return f"Opened {app_name} but its window did not appear within 10 seconds."

        # 2. Focus the window
        if not self._activate_window_obj(win):
            return f"Could not bring {app_name} to the foreground."
        time.sleep(1.5)  # let the app settle

        # 3. Type the text
        try:
            self._safe_type(text)
        except Exception as exc:
            logger.error("pyautogui typing failed in %s: %s", app_name, exc)
            return f"Failed to type in {app_name}: {exc}"

        logger.info("Typed '%s' in %s", text[:80], app_name)
        return f'\u2705 Typed "{text}" in {app_name}'

    # ── COMPOUND COMMAND ROUTER ───────────────────────────────────────────────

    BROWSER_NAMES = {"chrome", "brave", "edge", "firefox"}

    # Known browser executable paths for CLI tab launching
    _BROWSER_EXES: dict[str, list[str]] = {
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "google chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "brave": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
        "edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ],
    }

    def _find_browser_exe(self, browser: str) -> str | None:
        """Return the full path to a browser executable, or None if not found."""
        candidates = self._BROWSER_EXES.get(browser.lower(), [])
        for path in candidates:
            if Path(path).is_file():
                return path
        return None

    def _open_browser_tabs(self, browser: str, count: int) -> str:
        """Open `count` tabs in `browser` using CLI args (fast, no pyautogui).

        Chromium-based browsers (Chrome, Brave, Edge) accept multiple URL
        arguments and open each in its own tab. Firefox uses --new-tab.

        Args:
            browser: Browser name string.
            count: Number of tabs to open (already capped by caller).

        Returns:
            Result string.
        """
        exe = self._find_browser_exe(browser)
        if not exe:
            # Fall back to Start Menu launch + Ctrl+T loop for unknown browsers
            success = self.start_process(browser)
            if not success:
                return f"Could not find {browser} to open tabs."
            time.sleep(2)
            for _ in range(count - 1):
                pyautogui.hotkey("ctrl", "t")
                time.sleep(0.15)
            return f"Opened {browser} with {count} tabs (via keyboard shortcut)."

        try:
            is_firefox = "firefox" in browser.lower()
            if is_firefox:
                # Firefox: --new-window URL --new-tab URL --new-tab URL ...
                args = [exe, "--new-window", "about:blank"]
                for _ in range(count - 1):
                    args += ["--new-tab", "about:blank"]
            else:
                # Chromium: pass N URLs as positional args — each opens in a tab
                args = [exe, "--new-window"] + ["about:blank"] * count

            subprocess.Popen(args)
            logger.info("Opened %s with %d tabs via CLI: %s", browser, count, exe)
            return f"Opened {browser} with {count} tabs."
        except Exception as exc:
            return f"Failed to open {browser} tabs: {exc}"

    def _parse_compound_command(self, text: str) -> dict | None:
        """Detect and parse any compound (multi-step) command.

        Returns:
            Dict describing the compound action, or None if not compound.
        """
        nl = text.lower().strip()

        # 1. Messaging commands  (send/message/tell … on whatsapp/telegram)
        msg = self._parse_messaging_command(text)
        if msg:
            return {"type": "message", **msg}

        # 2. Browser navigation: "open chrome and go to youtube.com"
        m = re.search(
            r'(?:open|launch|start)\s+(\w+)\s+and\s+'
            r'(?:go\s+to|navigate\s+to|open|visit)\s+'
            r'((?:https?://)?[\w.\-/]+\.\w+[\w.\-/]*)',
            nl,
        )
        if m and m.group(1) in self.BROWSER_NAMES:
            url = m.group(2)
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            return {"type": "navigate", "app": m.group(1), "url": url}

        # 3. Browser search: "open chrome and search for python tutorials"
        m = re.search(
            r'(?:open|launch|start)\s+(\w+)\s+and\s+'
            r'(?:search|search\s+for|look\s+up|google)\s+(.+?)\s*$',
            nl,
        )
        if m and m.group(1) in self.BROWSER_NAMES:
            return {"type": "search", "app": m.group(1), "query": m.group(2).strip()}

        # 4. Open app and type text: "open notepad and type hello world"
        m = re.search(
            r'(?:open|launch|start)\s+(.+?)\s+and\s+'
            r'(?:type|write|enter|input)\s+(.+?)\s*$',
            nl,
        )
        if m:
            return {"type": "type", "app": m.group(1).strip(), "text": m.group(2).strip()}

        return None

    # ── SYSTEM CONTROL ────────────────────────────────────────────────────────

    def set_volume(self, level: float) -> bool:
        """Set the system volume.

        Args:
            level: Volume level from 0.0 to 1.0.

        Returns:
            True if volume was set successfully, False on failure.
        """
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level)), None)
            logger.info("Volume set to %.0f%%", level * 100)
            return True
        except Exception as exc:
            logger.error("Failed to set volume: %s", exc)
            return False

    def get_volume_level(self) -> float | None:
        """Return the current master volume scalar from 0.0 to 1.0."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            return float(volume.GetMasterVolumeLevelScalar())
        except Exception as exc:
            logger.error("Failed to get volume level: %s", exc)
            return None

    def run_command(self, command: str) -> tuple[str, str, int]:
        """Run a shell command and return output.

        Args:
            command: The shell command string.

        Returns:
            Tuple of (stdout, stderr, returncode).
        """
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out after 30 seconds", 1
        except Exception as exc:
            return "", str(exc), 1

    # ── MEDIA PLAYBACK ────────────────────────────────────────────────────────

    def _play_media(self, nl: str) -> str:
        """Play music/video by searching YouTube, finding the first video, and opening it.

        Opens Chrome with the configured user profile and auto-plays the video.

        Args:
            nl: Lowercase user input string.

        Returns:
            Confirmation string.
        """
        import webbrowser
        from urllib.parse import quote_plus

        import httpx

        # Extract the query: remove common filler words
        query = nl
        for filler in ("play ", "play me ", "play some ", "put on ", "songs play",
                        "song play", "songs", "song", "music", "on youtube",
                        "on spotify", "please", "for me", "that", "those",
                        "the", "some", "thst"):
            query = query.replace(filler, " ")
        query = " ".join(query.split()).strip()

        if not query:
            query = "top hits playlist"

        # Fetch YouTube search HTML and extract the first video ID
        try:
            resp = httpx.get(
                f"https://www.youtube.com/results?search_query={quote_plus(query + ' songs')}",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=10,
            )
            match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
            if match:
                video_id = match.group(1)
                video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
                self._open_in_chrome(video_url)
                logger.info("Playing media — opened YouTube video: %s (%s)", query, video_id)
                return f"Now playing '{query}' on YouTube."
        except Exception as exc:
            logger.warning("YouTube video lookup failed: %s — falling back to search page", exc)

        # Fallback: open search results page
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query + ' songs')}"
        self._open_in_chrome(search_url)
        return f"Playing '{query}' — opened YouTube search in your browser."

    def _open_in_chrome(self, url: str) -> None:
        """Open a URL in Chrome with the configured user profile.

        Tries to find Chrome and launch with the user's profile so
        they stay logged in. Falls back to default browser if Chrome not found.

        Args:
            url: The URL to open.
        """
        chrome_profile = self.config.get("browser", {}).get("chrome_profile", "")
        chrome_paths = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        ]

        chrome_exe = None
        for p in chrome_paths:
            if p.exists():
                chrome_exe = str(p)
                break

        if chrome_exe:
            cmd = [chrome_exe]
            if chrome_profile:
                cmd.append(f"--profile-directory={chrome_profile}")
            cmd.append(url)
            try:
                subprocess.Popen(cmd)
                return
            except Exception as exc:
                logger.warning("Chrome launch failed: %s — falling back to default browser", exc)

        # Fallback to default browser
        import webbrowser
        webbrowser.open(url)

    # ── NATURAL LANGUAGE HANDLER ──────────────────────────────────────────────

    async def execute(self, natural_language: str, context: dict) -> str:
        """Parse and execute a natural language OS command.

        Args:
            natural_language: The user's natural language request.
            context: Context dict from ContextBuilder.

        Returns:
            Confirmation string of what was done.
        """
        nl = natural_language.lower()
        for src, dst in self.COMMAND_SYNONYMS:
            nl = nl.replace(src, dst)

        # ── "type <text> in <app>" / "write <text> in <app>" ─────────────
        # Patterns:
        #   "type hello in notepad"
        #   "write hello world in notepad"
        #   "type i love you in notepad"
        m = re.search(
            r'(?:type|write)\s+(.+?)\s+in\s+(\w+)\s*$',
            nl, re.I,
        )
        if m:
            text = m.group(1).strip()
            app = m.group(2).strip()
            return await self.type_in_app(app, text)

        # "in notepad type hello"
        m = re.search(
            r'in\s+(\w+)\s+(?:type|write)\s+(.+?)\s*$',
            nl, re.I,
        )
        if m:
            app = m.group(1).strip()
            text = m.group(2).strip()
            return await self.type_in_app(app, text)

        # "open notepad and type hello" / "launch notepad and write hello"
        # (also handled by compound router, but this catches it earlier
        #  and routes through the more robust type_in_app)
        m = re.search(
            r'(?:open|launch|start)\s+(\w+)\s+and\s+(?:type|write)\s+(.+?)\s*$',
            nl, re.I,
        )
        if m:
            app = m.group(1).strip()
            text = m.group(2).strip()
            return await self.type_in_app(app, text)

        # ── Compound / multi-step commands (highest priority) ─────────────
        compound = self._parse_compound_command(natural_language)
        if compound:
            ctype = compound["type"]
            if ctype == "message":
                if compound["app"] == "whatsapp":
                    return await self._send_whatsapp_message(
                        compound["contact"], compound["message"],
                    )
                if compound["app"] == "telegram":
                    return await self._send_telegram_message(
                        compound["contact"], compound["message"],
                    )
            elif ctype == "navigate":
                return await self._open_and_navigate(
                    compound["app"], compound["url"],
                )
            elif ctype == "search":
                return await self._open_and_search_browser(
                    compound["app"], compound["query"],
                )
            elif ctype == "type":
                return await self._open_and_type(
                    compound["app"], compound["text"],
                )

        # ── "open N tabs/windows in APP" ──────────────────────────────────
        tab_match = re.search(
            r'open\s+(\d+)\s+(?:tabs?|windows?)\s+(?:in|on|with|of)\s+(.+)', nl,
        )
        if tab_match:
            count = min(int(tab_match.group(1)), 200)  # cap at 200
            app_name = tab_match.group(2).strip()
            return self._open_browser_tabs(app_name, count)

        # ── Simple single-step commands ───────────────────────────────────

        all_app_names = set(self.APP_ALIASES.keys()) | set(self._start_apps.keys())

        # ── Multi-app: "open chrome, spotify and discord"  ────────────────
        # Also handles mixed: "open chrome and spotify and take a screenshot"
        # Extract everything after open/launch/start, split on commas and "and"
        multi_open = re.search(
            r'\b(?:open|launch|start)\s+(.+)', nl,
        )
        if multi_open:
            raw = multi_open.group(1).strip()
            # Split on commas and the word "and"
            tokens = [t.strip() for t in re.split(r',\s*|\s+and\s+', raw) if t.strip()]

            # Separate known app names from other actions ("take a screenshot", etc.)
            found_apps = []
            other_actions = []
            for token in tokens:
                # Check if token matches a known app name (exact or substring)
                matched_app = None
                for app in sorted(all_app_names, key=len, reverse=True):
                    if app in token or token in app:
                        matched_app = app
                        break
                if matched_app:
                    found_apps.append(matched_app)
                else:
                    other_actions.append(token)

            if len(found_apps) > 1:
                # True multi-app launch — open all of them
                results = []
                for app in found_apps:
                    ok = self.start_process(app)
                    results.append(f"{'Opened' if ok else 'Failed: '}{app}")
                summary = ", ".join(results)
                # Handle any non-app actions in the same command
                for action in other_actions:
                    if "screenshot" in action:
                        path = self.take_screenshot()
                        summary += f", screenshot saved to {path}"
                return summary

        # ── Single app open (original fast path) ─────────────────────────
        for app_name in sorted(all_app_names, key=len, reverse=True):
            if f"open {app_name}" in nl or f"start {app_name}" in nl or f"launch {app_name}" in nl:
                success = self.start_process(app_name)
                return f"Opened {app_name}." if success else f"Failed to open {app_name}."

        # ── "minimize all windows" / "show desktop" ──────────────────────────
        if any(p in nl for p in ("minimize all", "minimize all screens",
                                  "minimise all", "minimise all screens",
                                  "show desktop", "hide all windows",
                                  "clear desktop")):
            pyautogui.hotkey("win", "d")
            return "All windows minimized — desktop shown."

        # ── "close all windows" ───────────────────────────────────────────────
        if re.search(r'\bclose\s+all\s+windows\b', nl):
            pyautogui.hotkey("win", "d")
            return "Minimized all windows. (Use 'close all <app>' to kill a specific app.)"

        if "kill " in nl or "close " in nl or "stop " in nl:
            for app_name in sorted(all_app_names, key=len, reverse=True):
                if app_name in nl:
                    count = self.kill_process(app_name)
                    if count == 0:
                        return f"{app_name} is not running."
                    return f"Killed {count} {app_name} process{'es' if count > 1 else ''}."

        if "screenshot" in nl:
            path = self.take_screenshot()
            return f"Screenshot saved to {path}."

        # ── Media controls (must come BEFORE play check) ──────────────────
        if "pause" in nl or nl in ("stop music", "stop the music", "stop song", "stop the song"):
            pyautogui.press("playpause")
            return "Toggled pause/play."

        if "resume" in nl or nl == "continue":
            pyautogui.press("playpause")
            return "Music resumed."

        if any(w in nl for w in ("next song", "skip song", "skip")):
            pyautogui.hotkey("alt", "right")
            return "Skipped to next song."

        if any(w in nl for w in ("previous song", "prev song", "go back")):
            pyautogui.hotkey("alt", "left")
            return "Went to previous song."

        # ── Media / music playback ────────────────────────────────────────
        if "play " in nl or "song" in nl or "music" in nl:
            return self._play_media(nl)

        if "unmute" in nl:
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(0, None)
                return "System unmuted."
            except Exception as exc:
                logger.error("Unmute failed: %s", exc)
                pyautogui.press("volumemute")
                return "System unmuted (fallback command sent)."

        if "mute" in nl:
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(1, None)
                return "System muted."
            except Exception as exc:
                logger.error("Mute failed: %s", exc)
                pyautogui.press("volumemute")
                return "System muted (fallback command sent)."

        if "increase volume" in nl or "decrease volume" in nl:
            current = self.get_volume_level()
            if current is None:
                pyautogui.press("volumeup" if "increase volume" in nl else "volumedown")
                return "Volume adjusted."
            delta = 0.1 if "increase volume" in nl else -0.1
            new_level = max(0.0, min(1.0, current + delta))
            if self.set_volume(new_level):
                return f"Volume set to {round(new_level * 100)}%."
            return "Failed to adjust volume. Check audio device."

        if "volume" in nl:
            match = re.search(r"(\d+)", nl)
            if match:
                requested = int(match.group(1))
                level = requested / 100.0
                actual = min(100, requested)   # clipped to 100 by set_volume
                success = self.set_volume(level)
                if success:
                    msg = f"Volume set to {actual}%."
                    if requested > 100:
                        msg += f" (requested {requested}% — capped at 100%)"
                    return msg
                return f"Failed to set volume to {actual}%. Check audio device."

        if "battery" in nl or "charge" in nl or "power" in nl:
            battery = psutil.sensors_battery()
            if battery is None:
                return "No battery detected — this might be a desktop PC."
            percent = battery.percent
            plugged = battery.power_plugged
            secs_left = battery.secsleft
            status = "Plugged in (charging)" if plugged else "On battery"
            if secs_left == psutil.POWER_TIME_UNLIMITED:
                time_str = "Unlimited (plugged in)"
            elif secs_left == psutil.POWER_TIME_UNKNOWN or secs_left < 0:
                time_str = "Unknown"
            else:
                hours, remainder = divmod(secs_left, 3600)
                mins = remainder // 60
                time_str = f"{int(hours)}h {int(mins)}m"
            return (
                f"Battery: {percent}%\n"
                f"Status: {status}\n"
                f"Time remaining: {time_str}"
            )

        if any(w in nl for w in ("cpu", "eating", "process", "apps are running",
                                   "running apps", "running processes", "what is running",
                                   "what apps", "list processes", "show processes")):
            procs = self.list_processes()[:10]
            lines = [f"  {p['name']}: {p['cpu_percent']}% CPU, {p['memory_mb']}MB RAM" for p in procs]
            return "Running processes:\n" + "\n".join(lines)

        if "vram" in nl or "gpu" in nl:
            info = self.get_gpu_info()
            return (
                f"GPU: {info['gpu_name']}\n"
                f"VRAM: {info['gpu_vram_used_mb']}MB / {info['gpu_vram_total_mb']}MB\n"
                f"Load: {info['gpu_load_percent']}%"
            )

        if "run " in nl:
            match = re.search(r"run\s+(.+)", nl)
            if match:
                cmd = match.group(1).strip()
                stdout, stderr, code = self.run_command(cmd)
                if code == 0:
                    return f"Command executed successfully.\n{stdout[:500]}" if stdout else "Command executed successfully."
                return f"Command failed (code {code}).\n{stderr[:500]}"

        if "window" in nl:
            windows = self.list_windows()
            return "Open windows:\n" + "\n".join(f"  - {w}" for w in windows[:15])

        # ── Vision-guided click / GUI commands ─────────────────────────────
        if self.vision:
            # "click on X", "click the X button", "find and click X"
            click_match = re.search(
                r'(?:click\s+(?:on\s+)?(?:the\s+)?|find\s+and\s+click\s+(?:the\s+)?)(.+)',
                nl,
            )
            if click_match:
                target = click_match.group(1).strip().rstrip('.')
                ok, status = await self.vision.vision_click(target)
                return status

            # "type X into Y", "type X in the Y"
            type_match = re.search(
                r'type\s+["\']?(.+?)["\']?\s+(?:in(?:to)?)\s+(?:the\s+)?(.+)',
                nl,
            )
            if type_match:
                text = type_match.group(1).strip()
                target = type_match.group(2).strip().rstrip('.')
                ok, status = await self.vision.vision_type(target, text)
                return status

        # Fallback: try to find any app name mentioned after open/start/launch
        open_match = re.search(r'(?:open|start|launch|run)\s+([a-zA-Z][a-zA-Z0-9 ]+)', nl)
        if open_match:
            app_name = open_match.group(1).strip()
            success = self.start_process(app_name)
            if success:
                return f"Opened {app_name}."
            return f"Could not find or open '{app_name}'. Make sure it is installed."

        return f"I understood this as an OS command but couldn't determine the specific action. You said: '{natural_language}'"
