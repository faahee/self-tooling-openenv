"""Tool Registry — SQLite-backed registry with semantic search for synthesized tools."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Persistent registry for all brain-generated tools.

    Stores tool metadata in SQLite and maintains an in-memory cache of
    description embeddings for fast cosine-similarity search.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS tools (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT NOT NULL,
        code_path TEXT NOT NULL,
        parameters TEXT NOT NULL,
        created_by TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 0.0,
        total_runs INTEGER DEFAULT 0,
        last_used TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        tags TEXT,
        description_embedding TEXT
    );
    """

    def __init__(self, config: dict, embedder) -> None:
        """Initialize the tool registry.

        Args:
            config: Full JARVIS configuration dictionary.
            embedder: Shared SentenceTransformer instance for embedding descriptions.
        """
        self.db_path = Path(config["brain"]["db_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._lock = threading.Lock()
        self._init_db()
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._load_embeddings()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new SQLite connection for the current thread."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the tools table if it doesn't exist."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(self.CREATE_TABLE_SQL)
                conn.commit()
            finally:
                conn.close()

    def _load_embeddings(self) -> None:
        """Load all active tool description embeddings into memory for fast search."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT name, description_embedding FROM tools WHERE is_active = 1"
            ).fetchall()
            for row in rows:
                if row["description_embedding"]:
                    emb = np.array(
                        json.loads(row["description_embedding"]), dtype=np.float32
                    )
                    self._embeddings_cache[row["name"]] = emb
        finally:
            conn.close()

    def register(
        self,
        name: str,
        description: str,
        code_path: str,
        parameters: dict,
        created_by: str = "agent",
        tags: list[str] | None = None,
        version: int = 1,
    ) -> int:
        """Register a new tool or update an existing one.

        Args:
            name: Unique tool name (snake_case).
            description: Human-readable description of what the tool does.
            code_path: Path to the tool's Python file.
            parameters: JSON-serializable parameter schema dict.
            created_by: "human" or "agent".
            tags: Optional list of tag strings.
            version: Tool version number.

        Returns:
            The tool's database ID.
        """
        # Combine tool name (as natural-language words) with description
        # so searches match both "what is my system uptime" and
        # "how long has my computer been running since last restart".
        name_words = name.replace("_", " ")
        embed_text = f"{name_words}: {description}"
        embedding = self.embedder.encode(embed_text)
        embedding_json = json.dumps(embedding.tolist())

        with self._lock:
            conn = self._get_conn()
            try:
                existing = conn.execute(
                    "SELECT id FROM tools WHERE name = ?", (name,)
                ).fetchone()

                if existing:
                    conn.execute(
                        """UPDATE tools SET description = ?, code_path = ?, parameters = ?,
                           version = ?, description_embedding = ?, is_active = 1, tags = ?
                           WHERE name = ?""",
                        (
                            description,
                            str(code_path),
                            json.dumps(parameters),
                            version,
                            embedding_json,
                            json.dumps(tags or []),
                            name,
                        ),
                    )
                    tool_id = existing["id"]
                else:
                    cursor = conn.execute(
                        """INSERT INTO tools (name, description, code_path, parameters,
                           created_by, version, description_embedding, tags)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            name,
                            description,
                            str(code_path),
                            json.dumps(parameters),
                            created_by,
                            version,
                            embedding_json,
                            json.dumps(tags or []),
                        ),
                    )
                    tool_id = cursor.lastrowid

                conn.commit()
                self._embeddings_cache[name] = embedding.astype(np.float32)
                return tool_id
            finally:
                conn.close()

    def search(self, query: str, threshold: float = 0.75) -> list[dict]:
        """Search tools by semantic similarity to a query.

        Args:
            query: Natural language task description.
            threshold: Minimum cosine similarity for a match.

        Returns:
            List of matching tool dicts sorted by similarity (highest first).
        """
        if not self._embeddings_cache:
            return []

        query_embedding = self.embedder.encode(query).astype(np.float32)
        results: list[dict] = []

        for name, emb in self._embeddings_cache.items():
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                continue
            similarity = float(np.dot(query_embedding, emb) / (norm_q * norm_e))
            if similarity >= threshold:
                tool = self.get_by_name(name)
                if tool:
                    tool["similarity"] = similarity
                    results.append(tool)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def get_by_name(self, name: str) -> dict | None:
        """Retrieve a tool by its unique name.

        Args:
            name: Tool name.

        Returns:
            Tool dict or None if not found/inactive.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM tools WHERE name = ? AND is_active = 1", (name,)
            ).fetchone()
            if row:
                result = dict(row)
                result.pop("description_embedding", None)
                return result
            return None
        finally:
            conn.close()

    def update_stats(self, name: str, success: bool) -> None:
        """Update run statistics for a tool after execution.

        Args:
            name: Tool name.
            success: Whether the last run succeeded.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                if success:
                    conn.execute(
                        """UPDATE tools SET
                           success_count = success_count + 1,
                           total_runs = total_runs + 1,
                           success_rate = CAST(success_count + 1 AS REAL) / (total_runs + 1),
                           last_used = ?
                           WHERE name = ?""",
                        (datetime.now().isoformat(), name),
                    )
                else:
                    conn.execute(
                        """UPDATE tools SET
                           failure_count = failure_count + 1,
                           total_runs = total_runs + 1,
                           success_rate = CAST(success_count AS REAL) / (total_runs + 1),
                           last_used = ?
                           WHERE name = ?""",
                        (datetime.now().isoformat(), name),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_all_active(self) -> list[dict]:
        """Return all active tools sorted by success rate descending.

        Returns:
            List of tool dicts.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM tools WHERE is_active = 1 ORDER BY success_rate DESC"
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d.pop("description_embedding", None)
                results.append(d)
            return results
        finally:
            conn.close()

    def get_version(self, name: str) -> int:
        """Get the current version number of a tool.

        Args:
            name: Tool name.

        Returns:
            Version number, or 0 if tool doesn't exist.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT version FROM tools WHERE name = ?", (name,)
            ).fetchone()
            return row["version"] if row else 0
        finally:
            conn.close()

    def deactivate(self, name: str) -> None:
        """Mark a tool as inactive (soft delete).

        Args:
            name: Tool name to deactivate.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE tools SET is_active = 0 WHERE name = ?", (name,)
                )
                conn.commit()
                self._embeddings_cache.pop(name, None)
            finally:
                conn.close()

    def set_success_rate(self, name: str, rate: float) -> None:
        """Manually set a tool's success rate (used for testing).

        Args:
            name: Tool name.
            rate: New success rate (0.0–1.0).
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE tools SET success_rate = ?, total_runs = MAX(total_runs, 10)
                       WHERE name = ?""",
                    (rate, name),
                )
                conn.commit()
            finally:
                conn.close()
