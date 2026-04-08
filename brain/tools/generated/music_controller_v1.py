"""Fully automated music controller for Apple Music, Spotify, and any active media player on Windows.

Supports: now_playing, play, pause, toggle, next, previous, stop, search.
Uses Windows media key simulation (works with any media app), window title
inspection for detecting the current track, and pywinauto UI automation
for searching and playing specific songs/artists in Apple Music.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import re
import time
import psutil

logger = logging.getLogger(__name__)

# ── Windows constants ────────────────────────────────────────────────
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# Process names we know how to read "now playing" from
_MEDIA_APPS: dict[str, list[str]] = {
    "apple_music": ["AppleMusic.exe", "Music.UI.exe"],
    "itunes":      ["iTunes.exe"],
    "spotify":     ["Spotify.exe"],
}

# Window titles that mean "idle / home screen" (not playing anything)
_IDLE_TITLES: set[str] = {
    "Apple Music", "iTunes", "Spotify", "Spotify Premium", "Spotify Free", "",
    "WinUI Desktop", "Pop-upHost", "Default IME", "MSCTFIME UI",
    "GDI+ Window", "GDI+ Window (AppleMusic.exe)",
}


def _send_media_key(vk: int) -> None:
    """Press and release a virtual media key."""
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


def _try_apple_music_transport(action: str) -> dict | None:
    """Use Apple Music's transport button when available for reliable play/pause."""
    if action not in {"play", "pause", "toggle"}:
        return None
    try:
        from pywinauto import Desktop
    except ImportError:
        return None

    try:
        am_pids: set[int] = set()
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] in ("AppleMusic.exe", "Music.UI.exe"):
                am_pids.add(proc.info["pid"])
        if not am_pids:
            return None

        desktop = Desktop(backend="uia")
        app_win = None
        for title in ("Apple Music",):
            try:
                candidate = desktop.window(title=title)
                _ = candidate.element_info.name
                app_win = candidate
                break
            except Exception:
                continue
        if app_win is None:
            return None

        btn = app_win.child_window(
            auto_id="TransportControl_PlayPauseStop", control_type="Button"
        )
        state = (btn.element_info.name or "").strip()

        if action == "pause":
            if state == "Play":
                return {"success": True, "result": "Apple Music is already paused", "error": None}
            btn.click_input()
            time.sleep(0.3)
            return {"success": True, "result": "Paused Apple Music", "error": None}

        if action == "play":
            if state == "Pause":
                return {"success": True, "result": "Apple Music is already playing", "error": None}
            btn.click_input()
            time.sleep(0.3)
            return {"success": True, "result": "Resumed Apple Music", "error": None}

        btn.click_input()
        time.sleep(0.3)
        return {"success": True, "result": "Toggled Apple Music playback", "error": None}
    except Exception as exc:
        logger.warning("Apple Music transport automation failed: %s", exc)
        return None


def _find_media_pids() -> dict[str, set[int]]:
    """Return {app_label: {pid, ...}} for running media players."""
    found: dict[str, set[int]] = {}
    for proc in psutil.process_iter(["pid", "name"]):
        name = proc.info["name"] or ""
        for label, exe_names in _MEDIA_APPS.items():
            if name in exe_names:
                found.setdefault(label, set()).add(proc.info["pid"])
    return found


def _get_window_titles(pids: set[int]) -> list[str]:
    """Return visible window titles belonging to the given PIDs."""
    user32 = ctypes.windll.user32
    titles: list[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lp):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                t = buf.value.strip()
                if t and t not in _IDLE_TITLES:
                    titles.append(t)
        return True

    user32.EnumWindows(_cb, 0)
    return titles


def _parse_now_playing(title: str, app: str) -> dict[str, str]:
    """Parse an 'Artist - Song' style window title."""
    # Apple Music / iTunes / Spotify all use "Artist — Title" or "Artist - Title"
    for sep in (" \u2014 ", " - "):
        if sep in title:
            parts = title.split(sep, 1)
            return {"artist": parts[0].strip(), "title": parts[1].strip(), "app": app}
    return {"title": title, "artist": "", "app": app}


def _get_apple_music_now_playing() -> dict | None:
    """Read now-playing info from Apple Music via UI Automation (LCD element).

    Returns dict with title/artist/app keys, or None if not playing.
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return None

    try:
        user32 = ctypes.windll.user32
        am_pids: set[int] = set()
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] in ("AppleMusic.exe", "Music.UI.exe"):
                am_pids.add(proc.info["pid"])
        if not am_pids:
            return None

        # Restore if minimized
        target_hwnd = None

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _find_am(hwnd, _lp):
            nonlocal target_hwnd
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in am_pids:
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    if title == "Apple Music" or "\u2014" in title:
                        target_hwnd = hwnd
                        return False
            return True
        user32.EnumWindows(_find_am, 0)

        # Try to read without restoring focus first (less disruptive)
        desktop = Desktop(backend="uia")
        app_win = None
        try:
            app_win = desktop.window(title="Apple Music")
            # Quick accessibility check — will throw if window isn't reachable
            _ = app_win.element_info.name
        except Exception:
            app_win = None

        if app_win is None and target_hwnd:
            # Window not reachable (likely minimized) — restore it
            user32.ShowWindow(target_hwnd, 9)
            time.sleep(0.3)
            user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.5)
            try:
                app_win = desktop.window(title="Apple Music")
            except Exception:
                return None

        if app_win is None:
            return None

        # Check transport button: if it says "Play" then nothing is playing
        try:
            btn = app_win.child_window(
                auto_id="TransportControl_PlayPauseStop", control_type="Button"
            )
            btn_name = btn.element_info.name
            logger.info("Apple Music transport button: %s", btn_name)
            if btn_name == "Play":
                return None  # Not playing
        except Exception as e:
            logger.warning("Could not read transport button: %s", e)

        # Read LCD element (auto_id='LCD') which contains now-playing info
        try:
            lcd = app_win.child_window(auto_id="LCD", control_type="Group")
            lcd_name = lcd.element_info.name or ""
            if lcd_name:
                # LCD text is like "Song Title Artist Name"
                # Try to find the ScrollingText children for structured data
                song_title = ""
                artist_info = ""
                for child in lcd.descendants():
                    try:
                        aid = child.element_info.automation_id or ""
                        if aid == "ScrollingText":
                            text = child.element_info.name or ""
                            if not song_title:
                                song_title = text
                            elif not artist_info:
                                artist_info = text
                    except Exception:
                        pass

                if song_title:
                    # artist_info may be "Artist — Song" or just "Artist"
                    artist = ""
                    if artist_info:
                        for sep in (" \u2014 ", " - "):
                            if sep in artist_info:
                                artist = artist_info.split(sep)[0].strip()
                                break
                        if not artist:
                            artist = artist_info
                    return {"title": song_title, "artist": artist, "app": "apple_music"}
        except Exception:
            pass

        return None
    except Exception as exc:
        logger.error("Apple Music UI now_playing failed: %s", exc)
        return None


def _search_and_play_apple_music(query: str) -> dict:
    """Search for music in Apple Music and play the first matching song.

    Uses pywinauto UI Automation to interact with the Apple Music window:
    1. Focus the Apple Music window
    2. Click the Search text box
    3. Type the search query
    4. Wait for results, right-click the first song
    5. Select 'Play' from the context menu
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"success": False, "result": None,
                "error": "pywinauto is not installed (pip install pywinauto)"}

    # Check Apple Music is running — launch it if not
    import subprocess as _sp
    am_pids: set[int] = set()
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] in ("AppleMusic.exe", "Music.UI.exe"):
            am_pids.add(proc.info["pid"])
    if not am_pids:
        try:
            _sp.Popen(["explorer.exe", "shell:appsFolder\\AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"])
            time.sleep(5)  # wait for Apple Music to fully launch
            for proc in psutil.process_iter(["pid", "name"]):
                if proc.info["name"] in ("AppleMusic.exe", "Music.UI.exe"):
                    am_pids.add(proc.info["pid"])
        except Exception as exc:
            return {"success": False, "result": None,
                    "error": f"Could not launch Apple Music: {exc}"}
    if not am_pids:
        return {"success": False, "result": None,
                "error": "Apple Music failed to start."}

    try:
        # Restore Apple Music window via ctypes (works even when minimized)
        user32 = ctypes.windll.user32
        target_hwnd = None

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _find_am(hwnd, _lp):
            nonlocal target_hwnd
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in am_pids:
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    if title == "Apple Music" or "\u2014" in title:
                        target_hwnd = hwnd
                        return False  # stop enumeration
            return True
        user32.EnumWindows(_find_am, 0)

        if target_hwnd:
            user32.ShowWindow(target_hwnd, 9)       # SW_RESTORE
            time.sleep(0.3)
            user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.8)

        desktop = Desktop(backend="uia")
        app_win = desktop.window(title="Apple Music")
        app_win.set_focus()
        time.sleep(0.4)

        # Click and fill the search box
        search_box = app_win.child_window(auto_id="TextBox", control_type="Edit")
        search_box.click_input()
        time.sleep(0.3)
        search_box.type_keys("^a", pause=0.05)   # select all
        time.sleep(0.1)
        # Escape pywinauto special characters in query to prevent crashes
        safe_query = query.replace('{', '{{').replace('}', '}}')
        for ch in ('+', '^', '%', '(', ')'):
            safe_query = safe_query.replace(ch, '{' + ch + '}')
        search_box.type_keys(safe_query, with_spaces=True, pause=0.02)
        time.sleep(0.5)
        search_box.type_keys("{ENTER}", pause=0.05)
        time.sleep(3)  # wait for search results

        # Find the first song in search results
        # Real songs look like "Sakkarakatti Song · Hiphop Tamizha" (with middle dot)
        # Headers are just "Songs", "Albums", etc.
        descendants = app_win.descendants()
        song_target = None
        for d in descendants:
            try:
                name = d.element_info.name or ""
                ct = d.element_info.control_type or ""
                if ct == "ListItem" and "\u00b7" in name:
                    # Middle dot separator = actual song (e.g. "Title · Artist")
                    song_target = d
                    break
            except Exception:
                pass

        if not song_target:
            # Fallback: look for ListItems containing query terms
            query_words = [w.lower() for w in query.split() if len(w) > 2]
            for d in descendants:
                try:
                    name = d.element_info.name or ""
                    ct = d.element_info.control_type or ""
                    if ct == "ListItem" and len(name) > 15:
                        name_lower = name.lower()
                        if any(w in name_lower for w in query_words):
                            song_target = d
                            break
                except Exception:
                    pass

        if not song_target:
            return {"success": False, "result": None,
                    "error": f"No songs found for '{query}' in Apple Music search results"}

        song_name = song_target.element_info.name

        # Right-click to get context menu
        song_target.click_input(button="right")
        time.sleep(1.2)

        # Context menu may be a separate popup window; search ALL desktop windows
        played = False
        for w in desktop.windows():
            try:
                for d in w.descendants():
                    try:
                        mi_name = d.element_info.name or ""
                        mi_ct = d.element_info.control_type or ""
                        if mi_ct == "MenuItem" and mi_name.lower().startswith("play ") \
                                and "playlist" not in mi_name.lower():
                            d.click_input()
                            played = True
                            break
                    except Exception:
                        pass
                if played:
                    break
            except Exception:
                pass

        if not played:
            # Fallback: try exact "Play" menu item
            for w in desktop.windows():
                try:
                    for d in w.descendants():
                        try:
                            mi_name = d.element_info.name or ""
                            mi_ct = d.element_info.control_type or ""
                            if mi_ct == "MenuItem" and mi_name.strip().lower() == "play":
                                d.click_input()
                                played = True
                                break
                        except Exception:
                            pass
                    if played:
                        break
                except Exception:
                    pass

        if not played:
            return {"success": False, "result": None,
                    "error": f"Found '{song_name}' but could not play it from context menu"}

        time.sleep(1.5)

        # Verify playback by checking transport button
        try:
            play_btn = app_win.child_window(
                auto_id="TransportControl_PlayPauseStop", control_type="Button"
            )
            if play_btn.element_info.name == "Pause":
                # "Pause" means music IS playing
                return {"success": True,
                        "result": f"Now playing: {song_name} on Apple Music",
                        "error": None}
        except Exception:
            pass

        return {"success": True,
                "result": f"Started playing: {song_name} on Apple Music",
                "error": None}

    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ── Public API ───────────────────────────────────────────────────────

def music_controller(action: str = "now_playing", query: str = "") -> dict:
    """Control music playback or detect what is currently playing.

    Args:
        action: One of 'now_playing', 'play', 'pause', 'toggle' (play/pause),
                'next', 'previous', 'stop', 'search'.
        query: Search term for 'search' action (artist, song, or album name).

    Returns:
        dict: {"success": bool, "result": any, "error": str or None}
    """
    action = action.lower().strip()

    # ── Infer action from natural language if not a direct command ────
    _ACTION_KEYWORDS = {
        "play":     ["play", "resume", "start playing", "unpause"],
        "pause":    ["pause", "hold"],
        "toggle":   ["toggle"],
        "next":     ["next", "skip", "forward"],
        "previous": ["previous", "prev", "back", "last song", "go back"],
        "stop":     ["stop"],
        "now_playing": ["what song", "what's playing", "currently playing", "now playing",
                        "what is playing", "which song", "what track"],
    }

    # Detect search intent: if there's a specific artist/song name in the action
    _SEARCH_PATTERNS = ["play ", "listen to ", "search ", "find ", "put on "]
    _GENERIC_PLAY = {"play", "play music", "resume", "resume music", "start playing"}

    # If action is a known simple command but query is provided, use search
    if action in {"play", "search"} and query:
        action = "search"
    elif action not in {"play", "pause", "toggle", "next", "previous", "prev",
                        "stop", "now_playing", "search"}:
        # Check if this is a search request (e.g. "play hip hop tamizha")
        for pat in _SEARCH_PATTERNS:
            if pat in action:
                # Extract the query part after the keyword
                idx = action.index(pat) + len(pat)
                extracted = action[idx:].strip()
                # Filter out generic phrases that don't indicate specific music
                _NOISE_PHRASES = [
                    r"\bin apple music\b", r"\bon apple music\b",
                    r"\bin spotify\b", r"\bon spotify\b",
                ]
                _NOISE_WORDS = [
                    r"\bmusic\b", r"\bsongs?\b", r"\bsomething\b",
                    r"\btracks?\b",
                ]
                cleaned = extracted
                for noise_pat in _NOISE_PHRASES + _NOISE_WORDS:
                    cleaned = re.sub(noise_pat, "", cleaned, flags=re.IGNORECASE).strip()
                if cleaned and cleaned not in _GENERIC_PLAY:
                    query = query or cleaned
                    action = "search"
                    break

        if action != "search":
            # Try standard action keywords
            for act, keywords in _ACTION_KEYWORDS.items():
                if any(kw in action for kw in keywords):
                    action = act
                    break
            else:
                action = "now_playing"

    # If action is "search" but no query, fall back to play
    if action == "search" and not query:
        action = "play"

    # ── Search and play specific music in Apple Music ────────────
    if action == "search":
        return _search_and_play_apple_music(query)

    # ── Playback controls (work for ANY active media player) ─────
    control_map = {
        "play":     VK_MEDIA_PLAY_PAUSE,
        "pause":    VK_MEDIA_PLAY_PAUSE,
        "toggle":   VK_MEDIA_PLAY_PAUSE,
        "next":     VK_MEDIA_NEXT_TRACK,
        "previous": VK_MEDIA_PREV_TRACK,
        "prev":     VK_MEDIA_PREV_TRACK,
        "stop":     VK_MEDIA_STOP,
    }

    if action in control_map:
        try:
            # If no media player is running, launch Apple Music first
            if not _find_media_pids():
                import subprocess as _sp
                _sp.Popen(["explorer.exe", "shell:appsFolder\\AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"])
                time.sleep(5)
            app_specific = _try_apple_music_transport(action)
            if app_specific is not None:
                return app_specific
            _send_media_key(control_map[action])
            return {
                "success": True,
                "result": f"Media key '{action}' sent successfully",
                "error": None,
            }
        except Exception as exc:
            return {"success": False, "result": None, "error": str(exc)}

    # ── Now playing detection ────────────────────────────────────
    if action == "now_playing":
        try:
            app_pids = _find_media_pids()
            if not app_pids:
                return {
                    "success": False,
                    "result": None,
                    "error": "No music player is running (checked Apple Music, iTunes, Spotify)",
                }

            # Check each running player for a "now playing" window title
            for app_label, pids in app_pids.items():
                titles = _get_window_titles(pids)
                if titles:
                    info = _parse_now_playing(titles[0], app_label)
                    if info.get("artist"):
                        msg = f"Now playing: {info['title']} by {info['artist']} ({info['app']})"
                    else:
                        msg = f"Now playing: {info['title']} ({info['app']})"
                    return {"success": True, "result": msg, "error": None}

            # Window title didn't show track info — try Apple Music UI automation
            if "apple_music" in app_pids:
                am_info = _get_apple_music_now_playing()
                if am_info:
                    if am_info.get("artist"):
                        msg = f"Now playing: {am_info['title']} by {am_info['artist']} ({am_info['app']})"
                    else:
                        msg = f"Now playing: {am_info['title']} ({am_info['app']})"
                    return {"success": True, "result": msg, "error": None}

            # Players are running but nothing detected
            running = ", ".join(app_pids.keys())
            return {
                "success": True,
                "result": f"Music player(s) open ({running}) but nothing is currently playing",
                "error": None,
            }
        except Exception as exc:
            return {"success": False, "result": None, "error": str(exc)}

    return {
        "success": False,
        "result": None,
        "error": f"Unknown action '{action}'. Use: now_playing, play, pause, toggle, next, previous, stop",
    }
