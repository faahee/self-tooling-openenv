def get_wifi_signal_strength() -> dict:
    """Get the current WiFi signal strength, SSID, and connection quality."""
    try:
        import subprocess
        r = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return {"success": False, "result": None, "error": f"netsh failed: {r.stderr.strip()}"}

        lines = r.stdout.splitlines()
        ssid = signal = state = radio = band = channel = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("SSID") and "BSSID" not in stripped:
                ssid = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Signal"):
                signal = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("State"):
                state = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Radio type"):
                radio = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Band"):
                band = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Channel"):
                channel = stripped.split(":", 1)[1].strip()

        if state and state.lower() != "connected":
            return {"success": False, "result": None, "error": f"WiFi not connected (state: {state})"}
        if not signal:
            return {"success": False, "result": None, "error": "No WiFi interface found — may be on Ethernet only"}

        # Determine quality label
        pct = int(signal.replace("%", ""))
        if pct >= 80:
            quality = "Excellent"
        elif pct >= 60:
            quality = "Good"
        elif pct >= 40:
            quality = "Fair"
        else:
            quality = "Weak"

        parts = [f"Network: {ssid}", f"Signal: {signal}", f"Quality: {quality}"]
        if radio:
            parts.append(f"Radio: {radio}")
        if band:
            parts.append(f"Band: {band}")
        if channel:
            parts.append(f"Channel: {channel}")

        return {"success": True, "result": " | ".join(parts), "error": None}

    except FileNotFoundError:
        return {"success": False, "result": None, "error": "netsh command not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "result": None, "error": "netsh command timed out"}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}