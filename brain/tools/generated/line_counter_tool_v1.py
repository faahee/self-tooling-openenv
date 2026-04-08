import os


def line_counter_tool(project_path: str = ".") -> dict:
    """
    Recursively counts Python files and lines of code in a project directory.
    Reports total file count, total lines, and the top 5 largest files by line count.

    Args:
        project_path: Path to the project directory to scan.

    Returns:
        dict with success, result (summary string), and error.
    """
    try:
        if not os.path.isdir(project_path):
            return {"success": False, "result": None, "error": f"Directory not found: {project_path}"}

        file_lines = []  # list of (relative_path, line_count)

        for root, dirs, files in os.walk(project_path):
            # Skip hidden dirs, __pycache__, .venv, .git, node_modules
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', '.venv', 'node_modules')]
            for fname in files:
                if fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                            count = sum(1 for _ in f)
                        rel = os.path.relpath(fpath, project_path)
                        file_lines.append((rel, count))
                    except (PermissionError, OSError):
                        pass

        if not file_lines:
            return {"success": True, "result": "No Python files found in " + project_path, "error": None}

        total_files = len(file_lines)
        total_lines = sum(c for _, c in file_lines)

        # Top 5 by line count
        top5 = sorted(file_lines, key=lambda x: x[1], reverse=True)[:5]
        top_str = " | ".join(f"{name} ({lines} lines)" for name, lines in top5)

        result = f"{total_files} Python files, {total_lines} total lines. Top 5 largest: {top_str}"
        return {"success": True, "result": result, "error": None}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}