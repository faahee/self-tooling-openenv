def get_largest_files(drive: str = "C:/", count: int = 5) -> dict:
    """Find the largest files on a drive by recursively scanning all directories."""
    try:
        import os
        from pathlib import Path

        root = Path(drive)
        if not root.exists():
            return {"success": False, "result": None, "error": f"Drive {drive} not found"}

        # Skip protected/system directories to avoid PermissionError
        skip_dirs = {"$Recycle.Bin", "System Volume Information", "Windows",
                     "Recovery", "ProgramData", "$WinREAgent"}

        files = []
        for dirpath, dirnames, filenames in os.walk(str(root)):
            # Prune protected dirs in-place
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                fp = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fp)
                    files.append((fp, size))
                except (PermissionError, OSError):
                    pass

        if not files:
            return {"success": False, "result": None, "error": f"No accessible files found on {drive}"}

        # Sort by size descending and take top N
        files.sort(key=lambda x: x[1], reverse=True)
        top = files[:count]

        def fmt_size(b):
            for unit in ("B", "KB", "MB", "GB"):
                if b < 1024:
                    return f"{b:.1f} {unit}"
                b /= 1024
            return f"{b:.1f} TB"

        lines = [f"{i+1}. {fp} ({fmt_size(size)})" for i, (fp, size) in enumerate(top)]
        summary = f"Top {len(top)} largest files on {drive}:\n" + "\n".join(lines)

        return {"success": True, "result": summary, "error": None}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
