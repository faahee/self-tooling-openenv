"""Generate a simple JARVIS architecture diagram PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "jarvis_architecture_diagram.pdf"


def add_box(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=9, wrap=True)


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#374151")
    ax.add_patch(arrow)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "JARVIS AI Agent Architecture", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.93, "Local Windows AI assistant with orchestrator, agents, memory, brain tools, and MCP integration",
            ha="center", va="center", fontsize=11)

    add_box(ax, 0.05, 0.79, 0.18, 0.11, "User Input", "Text\nVoice\nHTTP API", "#dbeafe")
    add_box(ax, 0.30, 0.77, 0.22, 0.15, "UI + Entry Layer", "main.py\nui/ui_layer.py\nvoice_listener.py\nFastAPI routes", "#e0f2fe")
    add_box(ax, 0.58, 0.75, 0.28, 0.19, "Orchestrator", "core/orchestrator.py\ncentral routing, retries,\nworkflow handoff, MCP handoff", "#dcfce7")

    add_box(ax, 0.08, 0.53, 0.18, 0.17, "Classifier + Context", "intent_classifier.py\ncontext_builder.py\nresponse_cache.py", "#fef3c7")
    add_box(ax, 0.33, 0.52, 0.18, 0.18, "LLM Core", "llm_core.py\nfast model\nreasoning model\ncode / vision layers", "#fde68a")
    add_box(ax, 0.58, 0.50, 0.30, 0.22, "Agent Layer", "os_agent.py\nfile_agent.py\nterminal_agent.py\nweb/data/email/api/scheduler\nworkflow_agent.py", "#fae8ff")

    add_box(ax, 0.07, 0.24, 0.22, 0.18, "Memory Layer", "memory_system.py\npersonal_model.py\ninteraction_logger.py\npersistent context + learning", "#ede9fe")
    add_box(ax, 0.37, 0.23, 0.20, 0.20, "Brain / Tool System", "decision_engine.py\ntool_registry.py\ntool_synthesizer.py\nreact_engine.py", "#fecaca")
    add_box(ax, 0.66, 0.22, 0.24, 0.22, "External Tool Bridge", "mcp_client.py\nWindows-MCP\nfilesystem / fetch / memory\nsqlite / duckduckgo / puppeteer", "#cffafe")

    add_box(ax, 0.08, 0.04, 0.20, 0.12, "Windows / Files / Shell", "apps, processes, mouse/keyboard,\nfilesystem, screenshots, terminal", "#f3f4f6")
    add_box(ax, 0.39, 0.04, 0.18, 0.12, "Local Data Stores", "SQLite\nJSON logs\nscreenshots\ncached outputs", "#f3f4f6")
    add_box(ax, 0.68, 0.04, 0.20, 0.12, "Web / APIs / MCP Servers", "HTTP APIs\nweb pages\ncommunity MCP tools\nWindows automation", "#f3f4f6")

    add_arrow(ax, 0.23, 0.845, 0.30, 0.845)
    add_arrow(ax, 0.52, 0.845, 0.58, 0.845)

    add_arrow(ax, 0.58, 0.78, 0.26, 0.61)
    add_arrow(ax, 0.58, 0.79, 0.51, 0.62)
    add_arrow(ax, 0.73, 0.75, 0.73, 0.72)

    add_arrow(ax, 0.18, 0.53, 0.18, 0.42)
    add_arrow(ax, 0.45, 0.52, 0.47, 0.43)
    add_arrow(ax, 0.73, 0.50, 0.78, 0.44)

    add_arrow(ax, 0.18, 0.24, 0.18, 0.16)
    add_arrow(ax, 0.47, 0.23, 0.48, 0.16)
    add_arrow(ax, 0.78, 0.22, 0.78, 0.16)

    fig.savefig(OUTPUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
