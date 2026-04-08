"""Interaction Logger — SQLite log for every JARVIS interaction."""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Logs every interaction to a local SQLite database.

    Thread-safe: uses a threading.Lock around all database writes.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS interactions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        user_input  TEXT NOT NULL,
        intent      TEXT,
        response    TEXT NOT NULL,
        duration_ms INTEGER,
        cache_hit   INTEGER DEFAULT 0,
        session_id  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_intent ON interactions(intent);
    CREATE TABLE IF NOT EXISTS tool_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        event_type  TEXT NOT NULL,
        tool_name   TEXT NOT NULL,
        details     TEXT,
        session_id  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_tool_event_type ON tool_events(event_type);
    """

    def __init__(self, config: dict) -> None:
        """Initialize the interaction logger.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.db_path = Path(config["memory"]["db_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection for the current thread."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the database tables if they don't exist."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(self.CREATE_TABLE_SQL)
                conn.commit()
            finally:
                conn.close()

    def log(
        self,
        user_input: str,
        response: str,
        intent: str,
        duration_ms: int,
        cache_hit: bool,
    ) -> None:
        """Log a single interaction.

        Args:
            user_input: The user's input text.
            response: JARVIS response text.
            intent: Classified intent label.
            duration_ms: Time taken in milliseconds.
            cache_hit: Whether the response came from cache.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO interactions
                       (timestamp, user_input, intent, response, duration_ms, cache_hit, session_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        user_input,
                        intent,
                        response,
                        duration_ms,
                        int(cache_hit),
                        self.session_id,
                    ),
                )
                conn.commit()
            except Exception as exc:
                logger.error("Failed to log interaction: %s", exc)
            finally:
                conn.close()

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return the last *n* interactions.

        Args:
            n: Number of recent interactions to return.

        Returns:
            List of interaction dicts ordered newest-first.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_by_intent(self, intent: str, limit: int = 20) -> list[dict]:
        """Return interactions filtered by intent.

        Args:
            intent: Intent label to filter.
            limit: Maximum results.

        Returns:
            List of interaction dicts.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE intent = ? ORDER BY id DESC LIMIT ?",
                (intent, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across user_input and response columns.

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            List of matching interaction dicts.
        """
        conn = self._get_conn()
        try:
            pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE user_input LIKE ? OR response LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (pattern, pattern, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def log_tool_event(
        self,
        event_type: str,
        tool_name: str,
        details: dict,
    ) -> None:
        """Log a brain tool lifecycle event.

        Args:
            event_type: Event type (tool_created, tool_used, synthesis_failed).
            tool_name: Name of the tool involved.
            details: Additional event details dict.
        """
        import json as _json

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO tool_events
                       (timestamp, event_type, tool_name, details, session_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        event_type,
                        tool_name,
                        _json.dumps(details, default=str),
                        self.session_id,
                    ),
                )
                conn.commit()
            except Exception as exc:
                logger.error("Failed to log tool event: %s", exc)
            finally:
                conn.close()
