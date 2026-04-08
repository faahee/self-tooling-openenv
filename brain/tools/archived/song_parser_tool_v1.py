import ctypes
import ctypes.wintypes as wintypes
import psutil


def song_parser_tool() -> dict:
    """Detect the currently playing song on Spotify by reading its window title.

    Spotify's main window title shows 'Artist - Song Title' when music is
    playing. When paused or on the home screen it shows just 'Spotify'.

    Returns:
        dict: {"success": bool, "result": str, "error": str or None}
    """
    try:
        # --- Find Spotify process IDs via psutil ---
        spotify_pids: set[int] = set()
        for proc in psutil.process_iter(["pid", "name"]):
            name = proc.info["name"] or ""
            if "spotify" in name.lower():
                spotify_pids.add(proc.info["pid"])

        if not spotify_pids:
            return {"success": False, "result": None,
                    "error": "Spotify is not running"}

        # --- Enumerate windows belonging to Spotify ---
        user32 = ctypes.windll.user32
        GetWindowTextW = user32.GetWindowTextW
        GetWindowTextLengthW = user32.GetWindowTextLengthW
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId
        IsWindowVisible = user32.IsWindowVisible

        spotify_titles: list[str] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _enum_cb(hwnd, _lp):
            pid = wintypes.DWORD()
            GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in spotify_pids and IsWindowVisible(hwnd):
                length = GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    if title and title not in ("Spotify", "Spotify Premium",
                                               "Spotify Free", ""):
                        spotify_titles.append(title)
            return True

        user32.EnumWindows(_enum_cb, 0)

        if spotify_titles:
            title = spotify_titles[0]
            if " - " in title:
                artist, song = title.split(" - ", 1)
                return {"success": True,
                        "result": f"Now playing: {song} by {artist}",
                        "error": None}
            return {"success": True, "result": f"Spotify: {title}",
                    "error": None}

        return {"success": True,
                "result": "Spotify is open but nothing is currently playing",
                "error": None}

    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}