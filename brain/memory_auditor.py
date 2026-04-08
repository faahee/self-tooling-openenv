"""Memory Auditor — Automated garbage collection for JARVIS memory systems.

Periodically scans tool registry, knowledge graph, indexed texts,
generated tool files, and interaction logs. Cleans stale/broken/orphan
entries and reports what it did.

Runs on:
- JARVIS startup (quick scan)
- Every 30 minutes (background asyncio task)
- On demand via voice: "jarvis clean your memory"
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Audit report ──────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    """Result of a single audit run."""

    audit_timestamp: str = ""
    duration_seconds: float = 0.0
    tools_deactivated: list[str] = field(default_factory=list)
    tools_archived: list[str] = field(default_factory=list)
    ghost_entries_removed: int = 0
    knowledge_graph_entries_removed: int = 0
    indexed_texts_removed: int = 0
    orphan_files_deleted: list[str] = field(default_factory=list)
    log_entries_compressed: int = 0
    total_space_freed_kb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dict.

        Returns:
            Report as a JSON-safe dictionary.
        """
        return asdict(self)

    @property
    def summary(self) -> str:
        """One-line summary for UI display.

        Returns:
            Human-readable summary string.
        """
        parts: list[str] = []
        if self.tools_deactivated:
            parts.append(f"{len(self.tools_deactivated)} tools deactivated")
        if self.tools_archived:
            parts.append(f"{len(self.tools_archived)} tools archived")
        if self.ghost_entries_removed:
            parts.append(f"{self.ghost_entries_removed} ghost entries removed")
        total_cleaned = (
            self.knowledge_graph_entries_removed + self.indexed_texts_removed
        )
        if total_cleaned:
            parts.append(f"{total_cleaned} entries cleaned")
        if self.orphan_files_deleted:
            parts.append(f"{len(self.orphan_files_deleted)} orphan files deleted")
        if self.log_entries_compressed:
            parts.append(f"{self.log_entries_compressed} log entries compressed")
        if self.total_space_freed_kb > 0:
            parts.append(f"{self.total_space_freed_kb:.1f} KB freed")
        if not parts:
            return "Memory clean — nothing to do"
        return ", ".join(parts)


# ── Backup helpers ────────────────────────────────────────────────────────────

def _backup_file(path: Path) -> Path | None:
    """Create a .backup copy of *path* before modification.

    Args:
        path: File to back up.

    Returns:
        Path to the backup file, or None if the source does not exist.
    """
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + ".backup")
    shutil.copy2(path, backup)
    return backup


def _restore_backup(path: Path) -> bool:
    """Restore *path* from its .backup copy.

    Args:
        path: Original file path.

    Returns:
        True if restored successfully.
    """
    backup = path.with_suffix(path.suffix + ".backup")
    if backup.exists():
        shutil.copy2(backup, path)
        backup.unlink()
        logger.info("Restored %s from backup", path.name)
        return True
    return False


def _remove_backup(path: Path) -> None:
    """Remove the .backup copy after a successful audit step.

    Args:
        path: Original file path whose backup should be removed.
    """
    backup = path.with_suffix(path.suffix + ".backup")
    if backup.exists():
        backup.unlink()


# ── Memory Auditor ────────────────────────────────────────────────────────────

class MemoryAuditor:
    """Automated garbage collector for JARVIS memory systems."""

    INTERVAL_SECONDS: int = 30 * 60  # 30 minutes

    def __init__(
        self,
        config: dict,
        tool_registry: Any | None = None,
        interaction_logger: Any | None = None,
        ui_layer: Any | None = None,
    ) -> None:
        """Initialize the memory auditor.

        Args:
            config: Full JARVIS configuration dictionary.
            tool_registry: ToolRegistry instance (optional).
            interaction_logger: InteractionLogger instance (optional).
            ui_layer: UILayer instance for report display (optional).
        """
        self.config = config
        self.tool_registry = tool_registry
        self.interaction_logger = interaction_logger
        self.ui = ui_layer

        brain_cfg = config.get("brain", {})
        self.tools_dir = Path(brain_cfg.get("tools_dir", "brain/tools/generated"))
        self.archived_dir = self.tools_dir.parent / "archived"
        self.brain_db_path = Path(brain_cfg.get("db_path", "data/brain.db"))

        memory_cfg = config.get("memory", {})
        self.knowledge_graph_path = Path(
            memory_cfg.get("graph_path", "data/knowledge_graph.json")
        )
        self.indexed_texts_path = Path("data/indexed_texts.json")
        self.interaction_db_path = Path(
            memory_cfg.get("db_path", "data/interactions.db")
        )

        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run an initial audit then start the periodic background loop."""
        logger.info("Memory Auditor: running startup scan...")
        try:
            report = await self.run_audit()
            self._show_report(report)
        except Exception as exc:
            logger.error("Memory Auditor startup scan failed: %s", exc)

        self._running = True
        self._task = asyncio.create_task(self._periodic_loop())
        logger.info("Memory Auditor: background task started (every %ds).", self.INTERVAL_SECONDS)

    async def stop(self) -> None:
        """Cancel the periodic background task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Memory Auditor stopped.")

    async def _periodic_loop(self) -> None:
        """Sleep-audit loop that runs every INTERVAL_SECONDS."""
        while self._running:
            try:
                await asyncio.sleep(self.INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                report = await self.run_audit()
                self._show_report(report)
            except Exception as exc:
                logger.error("Memory Auditor periodic run failed: %s", exc)

    # ── Main audit entry point ────────────────────────────────────────────────

    async def run_audit(self) -> AuditReport:
        """Execute all audit checks and return a report.

        Returns:
            AuditReport with all findings and actions taken.
        """
        report = AuditReport(audit_timestamp=datetime.now().isoformat())
        start = time.monotonic()

        loop = asyncio.get_event_loop()

        # All checks are I/O bound — run in executor to avoid blocking
        await loop.run_in_executor(None, self._check_tool_health, report)
        await loop.run_in_executor(None, self._check_knowledge_graph, report)
        await loop.run_in_executor(None, self._check_indexed_texts, report)
        await loop.run_in_executor(None, self._check_generated_tools_dir, report)
        await loop.run_in_executor(None, self._check_interaction_logs, report)

        report.duration_seconds = round(time.monotonic() - start, 3)

        self._log_audit_event(report)
        return report

    # ── CHECK 1: Tool Health Audit ────────────────────────────────────────────

    def _check_tool_health(self, report: AuditReport) -> None:
        """Audit tool registry for broken, stale, and ghost entries.

        Args:
            report: AuditReport to populate with findings.
        """
        if not self.brain_db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.brain_db_path), timeout=10)
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            logger.warning("Tool health audit: cannot open brain DB: %s", exc)
            return

        try:
            rows = conn.execute(
                "SELECT name, code_path, success_rate, total_runs, "
                "last_used, is_active, version FROM tools"
            ).fetchall()

            now = datetime.now()

            for row in rows:
                name = row["name"]
                code_path = Path(row["code_path"]) if row["code_path"] else None
                success_rate = row["success_rate"] or 0.0
                total_runs = row["total_runs"] or 0
                last_used = row["last_used"]
                is_active = row["is_active"]
                version = row["version"] or 1

                # Ghost entry: file missing but registered
                if code_path and not code_path.exists():
                    conn.execute("DELETE FROM tools WHERE name = ?", (name,))
                    report.ghost_entries_removed += 1
                    logger.info("Removed ghost entry: %s", name)
                    continue

                if not is_active:
                    continue

                # Low success rate
                if success_rate < 0.3 and total_runs > 5:
                    conn.execute(
                        "UPDATE tools SET is_active = 0 WHERE name = ?", (name,)
                    )
                    report.tools_deactivated.append(name)
                    logger.info(
                        "Deactivated %s: success rate %.0f%%",
                        name,
                        success_rate * 100,
                    )
                    continue

                # Unused for 30+ days
                if last_used:
                    try:
                        last_dt = datetime.fromisoformat(last_used)
                        if (now - last_dt) > timedelta(days=30) and total_runs < 3:
                            conn.execute(
                                "UPDATE tools SET is_active = 0 WHERE name = ?",
                                (name,),
                            )
                            report.tools_archived.append(name)
                            logger.info(
                                "Archived %s: unused for 30+ days", name
                            )
                            continue
                    except (ValueError, TypeError):
                        pass

                # Old version files
                if version > 1 and code_path:
                    for old_v in range(1, version):
                        old_file = code_path.parent / f"{code_path.stem}_v{old_v}.py"
                        if old_file.exists():
                            size = old_file.stat().st_size
                            old_file.unlink()
                            report.total_space_freed_kb += size / 1024
                            logger.info(
                                "Cleaned old version: %s_v%d", name, old_v
                            )

            conn.commit()
        except Exception as exc:
            logger.error("Tool health audit failed: %s", exc)
        finally:
            conn.close()

    # ── CHECK 2: Knowledge Graph Audit ────────────────────────────────────────

    def _check_knowledge_graph(self, report: AuditReport) -> None:
        """Audit knowledge_graph.json for dead refs, hallucinations, dupes.

        Args:
            report: AuditReport to populate with findings.
        """
        if not self.knowledge_graph_path.exists():
            return

        backup = _backup_file(self.knowledge_graph_path)
        try:
            raw = self.knowledge_graph_path.read_text(encoding="utf-8")
            graph = json.loads(raw)
            original_size = self.knowledge_graph_path.stat().st_size

            # Gather deactivated tool names for reference checks
            deactivated_tools: set[str] = set()
            if self.brain_db_path.exists():
                try:
                    conn = sqlite3.connect(str(self.brain_db_path), timeout=10)
                    rows = conn.execute(
                        "SELECT name FROM tools WHERE is_active = 0"
                    ).fetchall()
                    deactivated_tools = {r[0] for r in rows}
                    conn.close()
                except Exception:
                    pass

            nodes: list[dict] = graph.get("nodes", [])
            cleaned_nodes: list[dict] = []
            quarantine: list[dict] = []
            seen_keys: set[str] = set()
            removed = 0

            for node in nodes:
                node_id = node.get("id", "")
                text = node.get("text", "")

                # Duplicate check (same id)
                if node_id in seen_keys:
                    removed += 1
                    continue
                seen_keys.add(node_id)

                # References a deactivated tool
                if deactivated_tools and any(
                    tool_name in text for tool_name in deactivated_tools
                ):
                    removed += 1
                    continue

                # No timestamp → likely hallucination
                if not node.get("created") and not node.get("timestamp"):
                    quarantine.append(node)
                    removed += 1
                    continue

                cleaned_nodes.append(node)

            # Also deduplicate links
            links = graph.get("links", [])
            valid_ids = {n.get("id") for n in cleaned_nodes}
            cleaned_links = [
                link
                for link in links
                if link.get("source") in valid_ids
                and link.get("target") in valid_ids
            ]

            # Store quarantined nodes in a separate section
            graph["nodes"] = cleaned_nodes
            graph["links"] = cleaned_links
            if quarantine:
                existing_q = graph.get("quarantine", [])
                existing_q.extend(quarantine)
                graph["quarantine"] = existing_q

            self.knowledge_graph_path.write_text(
                json.dumps(graph, indent=2, default=str), encoding="utf-8"
            )

            new_size = self.knowledge_graph_path.stat().st_size
            report.knowledge_graph_entries_removed = removed
            report.total_space_freed_kb += max(0, original_size - new_size) / 1024

            _remove_backup(self.knowledge_graph_path)

        except Exception as exc:
            logger.error("Knowledge graph audit failed: %s — restoring backup", exc)
            if backup:
                _restore_backup(self.knowledge_graph_path)

    # ── CHECK 3: Indexed Texts Audit ──────────────────────────────────────────

    def _check_indexed_texts(self, report: AuditReport) -> None:
        """Audit indexed_texts.json for hallucinations, stale, and empty entries.

        Args:
            report: AuditReport to populate with findings.
        """
        if not self.indexed_texts_path.exists():
            return

        backup = _backup_file(self.indexed_texts_path)
        try:
            raw = self.indexed_texts_path.read_text(encoding="utf-8")
            entries: list[dict] = json.loads(raw)
            original_size = self.indexed_texts_path.stat().st_size

            # Gather active tool function names to detect hallucination artifacts
            tool_func_names: set[str] = set()
            if self.brain_db_path.exists():
                try:
                    conn = sqlite3.connect(str(self.brain_db_path), timeout=10)
                    rows = conn.execute("SELECT name FROM tools").fetchall()
                    tool_func_names = {r[0] for r in rows}
                    conn.close()
                except Exception:
                    pass

            cutoff = datetime.now() - timedelta(days=90)
            cleaned: list[dict] = []
            removed = 0

            for entry in entries:
                text = entry.get("text", "")
                node_id = entry.get("node_id", "")

                # Empty or whitespace only
                if not text or not text.strip():
                    removed += 1
                    continue

                # Tool function name masquerading as response (hallucination)
                if tool_func_names:
                    stripped = text.strip()
                    # If the entire text is just a function name
                    if stripped in tool_func_names:
                        removed += 1
                        continue
                    # If text starts with "def function_name("
                    if any(
                        stripped.startswith(f"def {fn}(")
                        for fn in tool_func_names
                    ):
                        removed += 1
                        continue

                # Entry older than 90 days (parse from node_id timestamp)
                if node_id:
                    try:
                        # node_id format: mem_ID_YYYYMMDD_HHMMSS
                        parts = node_id.split("_")
                        if len(parts) >= 4:
                            date_str = parts[-2] + parts[-1]
                            entry_dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                            if entry_dt < cutoff:
                                removed += 1
                                continue
                    except (ValueError, IndexError):
                        pass

                cleaned.append(entry)

            self.indexed_texts_path.write_text(
                json.dumps(cleaned, indent=2, default=str), encoding="utf-8"
            )

            new_size = self.indexed_texts_path.stat().st_size
            report.indexed_texts_removed = removed
            report.total_space_freed_kb += max(0, original_size - new_size) / 1024

            _remove_backup(self.indexed_texts_path)

        except Exception as exc:
            logger.error(
                "Indexed texts audit failed: %s — restoring backup", exc
            )
            if backup:
                _restore_backup(self.indexed_texts_path)

    # ── CHECK 4: Generated Tools Directory Audit ──────────────────────────────

    def _check_generated_tools_dir(self, report: AuditReport) -> None:
        """Audit brain/tools/generated/ for orphan and deactivated tool files.

        Args:
            report: AuditReport to populate with findings.
        """
        if not self.tools_dir.exists():
            return

        # Get registered tool code_paths from DB
        registered_paths: set[str] = set()
        deactivated_paths: set[str] = set()
        if self.brain_db_path.exists():
            try:
                conn = sqlite3.connect(str(self.brain_db_path), timeout=10)
                rows = conn.execute(
                    "SELECT code_path, is_active FROM tools"
                ).fetchall()
                for code_path, is_active in rows:
                    resolved = str(Path(code_path).resolve())
                    if is_active:
                        registered_paths.add(resolved)
                    else:
                        deactivated_paths.add(resolved)
                conn.close()
            except Exception as exc:
                logger.warning(
                    "Generated tools audit: cannot read brain DB: %s", exc
                )
                return

        self.archived_dir.mkdir(parents=True, exist_ok=True)

        for py_file in self.tools_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            resolved = str(py_file.resolve())
            size = py_file.stat().st_size

            if resolved in deactivated_paths:
                # Move deactivated tool to archived/
                dest = self.archived_dir / py_file.name
                shutil.move(str(py_file), str(dest))
                report.tools_archived.append(py_file.name)
                report.total_space_freed_kb += size / 1024
                logger.info("Archived deactivated tool file: %s", py_file.name)

            elif resolved not in registered_paths:
                # Orphan file — not in registry at all
                py_file.unlink()
                report.orphan_files_deleted.append(py_file.name)
                report.total_space_freed_kb += size / 1024
                logger.info("Deleted orphan file: %s", py_file.name)

    # ── CHECK 5: Interaction Log Audit ────────────────────────────────────────

    def _check_interaction_logs(self, report: AuditReport) -> None:
        """Compress old interaction log entries into weekly summaries.

        Keeps the last 1000 raw entries. Entries older than 7 days are
        grouped by ISO week and compressed into a summary row.

        Args:
            report: AuditReport to populate with findings.
        """
        if not self.interaction_db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.interaction_db_path), timeout=10)
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            logger.warning("Interaction log audit: cannot open DB: %s", exc)
            return

        try:
            # Ensure summary table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interaction_summaries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_label  TEXT NOT NULL,
                    total_count INTEGER NOT NULL,
                    intents     TEXT,
                    created_at  TEXT NOT NULL
                )
            """)

            # Count total entries
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            if total <= 1000:
                conn.close()
                return

            # Find the 1000th-newest entry's id
            keep_boundary = conn.execute(
                "SELECT id FROM interactions ORDER BY id DESC LIMIT 1 OFFSET 999"
            ).fetchone()
            if not keep_boundary:
                conn.close()
                return

            cutoff_id = keep_boundary[0]

            # Also enforce 7-day minimum age for compression
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()

            # Get old entries to compress
            old_rows = conn.execute(
                """SELECT id, timestamp, intent FROM interactions
                   WHERE id <= ? AND timestamp < ?
                   ORDER BY timestamp""",
                (cutoff_id, seven_days_ago),
            ).fetchall()

            if not old_rows:
                conn.close()
                return

            # Group by ISO week
            weekly: dict[str, dict[str, Any]] = {}
            ids_to_delete: list[int] = []
            for row in old_rows:
                ids_to_delete.append(row["id"])
                ts = row["timestamp"]
                intent = row["intent"] or "UNKNOWN"
                try:
                    dt = datetime.fromisoformat(ts)
                    week_label = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
                except (ValueError, TypeError):
                    week_label = "unknown-week"

                if week_label not in weekly:
                    weekly[week_label] = {"count": 0, "intents": {}}
                weekly[week_label]["count"] += 1
                weekly[week_label]["intents"][intent] = (
                    weekly[week_label]["intents"].get(intent, 0) + 1
                )

            # Insert weekly summaries
            for week_label, data in weekly.items():
                conn.execute(
                    """INSERT INTO interaction_summaries
                       (week_label, total_count, intents, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        week_label,
                        data["count"],
                        json.dumps(data["intents"]),
                        datetime.now().isoformat(),
                    ),
                )

            # Delete compressed raw entries in batches
            batch_size = 500
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i : i + batch_size]
                placeholders = ",".join("?" * len(batch))
                conn.execute(
                    f"DELETE FROM interactions WHERE id IN ({placeholders})",
                    batch,
                )

            conn.commit()
            report.log_entries_compressed = len(ids_to_delete)
            logger.info(
                "Compressed %d interaction log entries into %d weekly summaries",
                len(ids_to_delete),
                len(weekly),
            )

        except Exception as exc:
            logger.error("Interaction log audit failed: %s", exc)
        finally:
            conn.close()

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _show_report(self, report: AuditReport) -> None:
        """Display the audit report via UILayer.

        Args:
            report: Completed AuditReport.
        """
        if self.ui:
            self.ui.show_audit_report(report)
        else:
            logger.info("Memory audit complete: %s", report.summary)

    def _log_audit_event(self, report: AuditReport) -> None:
        """Log the audit event to InteractionLogger.

        Args:
            report: Completed AuditReport.
        """
        if not self.interaction_logger:
            return
        try:
            self.interaction_logger.log_tool_event(
                "memory_audit",
                "memory_auditor",
                report.to_dict(),
            )
        except Exception as exc:
            logger.debug("Failed to log audit event: %s", exc)
