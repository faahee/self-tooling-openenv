"""Checkpoint Manager — Save and restore file states for undo support."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A single saved file state."""

    filepath: str
    content: str | None  # None = file did not exist before the edit
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )
    description: str = ""


class CheckpointManager:
    """Saves file states before edits so any change can be undone.

    Works like a stack — undo always reverts the most recent checkpoint.
    Stores up to MAX_CHECKPOINTS entries (oldest are dropped automatically).
    """

    MAX_CHECKPOINTS = 30

    def __init__(self) -> None:
        self._stack: deque[Checkpoint] = deque(maxlen=self.MAX_CHECKPOINTS)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def snapshot(self, filepath: str, description: str = "") -> None:
        """Save the current state of a file before modifying it.

        If the file does not yet exist, records that so undo can delete it.

        Args:
            filepath: Path to the file to snapshot.
            description: Optional human-readable description of the edit.
        """
        p = Path(filepath)
        content: str | None = None
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Could not snapshot %s: %s", filepath, exc)
                return  # Don't save a corrupt checkpoint
        self._stack.append(
            Checkpoint(
                filepath=str(p.resolve()),
                content=content,
                description=description,
            )
        )
        logger.debug("Checkpoint saved: %s (%s)", filepath, description or "no description")

    def undo(self) -> str:
        """Restore the most recently snapshotted file.

        Returns:
            Human-readable description of what was undone.
        """
        if not self._stack:
            return "Nothing to undo — no checkpoints saved."

        cp = self._stack.pop()
        p = Path(cp.filepath)
        try:
            if cp.content is None:
                # File was brand new — undo by deleting it
                if p.exists():
                    p.unlink()
                    return f"Undone: deleted newly created file\n  {cp.filepath}"
                return f"Undo: {cp.filepath} was already gone."
            # File existed before — restore original content
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(cp.content, encoding="utf-8")
            desc = f" ({cp.description})" if cp.description else ""
            return f"Undone{desc}: restored {cp.filepath} to state from {cp.timestamp}"
        except Exception as exc:
            return f"Undo failed for {cp.filepath}: {exc}"

    def list_checkpoints(self) -> str:
        """Return a formatted list of all saved checkpoints.

        Returns:
            Human-readable checkpoint list string.
        """
        if not self._stack:
            return "No checkpoints saved yet."
        lines = [
            f"  {i + 1}. [{cp.timestamp}] {cp.filepath}"
            + (f"  — {cp.description}" if cp.description else "")
            for i, cp in enumerate(self._stack)
        ]
        return f"Checkpoints ({len(self._stack)} saved, most recent last):\n" + "\n".join(lines)

    @property
    def has_checkpoints(self) -> bool:
        """True if there is at least one checkpoint to undo."""
        return len(self._stack) > 0
