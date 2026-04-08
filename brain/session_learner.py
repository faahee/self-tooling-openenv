"""Session Learner — Extracts lessons from each session and persists them.

Reads interaction logs and tool events, identifies failure/success patterns,
writes structured lessons to personal_model.json, and generates a human-
readable session summary.

Triggers:
- On JARVIS shutdown (always)
- On demand: "jarvis what did you learn today"
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum lessons injected into context per session
MAX_INJECTED_LESSONS = 5


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Lesson:
    """A single learned pattern."""

    pattern_type: str  # A, B, C, or D
    learned_at: str = ""
    session_evidence: str = ""
    lesson: str = ""
    confidence: float = 0.5
    times_reinforced: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            Dict representation.
        """
        return asdict(self)


@dataclass
class SessionSummary:
    """Human-readable session summary."""

    date: str = ""
    duration_minutes: float = 0.0
    tasks_completed: int = 0
    new_tools_created: int = 0
    tools_failed: int = 0
    lessons_learned: int = 0
    top_lesson: str = ""
    intent_breakdown: dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        """Format for terminal and file output.

        Returns:
            Multi-line summary string.
        """
        lines = [
            f"JARVIS Session Summary -- {self.date}",
            f"  Duration:          {self.duration_minutes:.0f} minutes",
            f"  Tasks completed:   {self.tasks_completed}",
            f"  New tools created: {self.new_tools_created}",
            f"  Tools failed:      {self.tools_failed}",
            f"  Lessons learned:   {self.lessons_learned}",
        ]
        if self.intent_breakdown:
            top_intents = sorted(
                self.intent_breakdown.items(), key=lambda x: x[1], reverse=True
            )[:5]
            intent_str = ", ".join(f"{k}: {v}" for k, v in top_intents)
            lines.append(f"  Top intents:       {intent_str}")
        if self.top_lesson:
            lines.append(f"  Top lesson:        {self.top_lesson}")
        return "\n".join(lines)


# ── Session Learner ──────────────────────────────────────────────────────────

class SessionLearner:
    """Extracts patterns from the current session and writes lessons."""

    def __init__(
        self,
        config: dict,
        interaction_logger: Any,
        personal_model: Any,
        ui_layer: Any | None = None,
    ) -> None:
        """Initialize the session learner.

        Args:
            config: Full JARVIS configuration dictionary.
            interaction_logger: InteractionLogger instance.
            personal_model: PersonalModel instance.
            ui_layer: Optional UILayer for display.
        """
        self.config = config
        self.interaction_logger = interaction_logger
        self.personal_model = personal_model
        self.ui = ui_layer

        memory_cfg = config.get("memory", {})
        self.interaction_db = Path(memory_cfg.get("db_path", "data/interactions.db"))
        self.personal_model_path = Path(
            memory_cfg.get("personal_model_path", "data/personal_model.json")
        )
        self.brain_db_path = Path(
            config.get("brain", {}).get("db_path", "data/brain.db")
        )

        self.summaries_dir = Path("data/session_summaries")
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

    # ── Main entry points ─────────────────────────────────────────────────────

    async def run_shutdown_learning(self) -> SessionSummary:
        """Full learning pipeline — called on JARVIS shutdown.

        Must complete within 30 seconds.

        Returns:
            SessionSummary of this session.
        """
        start = time.monotonic()
        logger.info("Session Learner: starting shutdown analysis...")

        # Step 1: Read session data
        session_data = self._read_session_data()

        # Step 2: Extract patterns
        lessons = self._extract_patterns(session_data)

        # Step 3: Write lessons to personal model
        new_count = self._write_lessons(lessons)

        # Step 4: Build summary
        summary = self._build_summary(session_data, new_count)

        # Step 5: Save summary to file
        self._save_summary(summary)

        elapsed = time.monotonic() - start
        logger.info(
            "Session Learner: completed in %.1fs — %d lessons learned",
            elapsed,
            new_count,
        )

        # Show in UI
        if self.ui:
            self.ui.show_session_summary(summary)

        return summary

    async def get_lessons_learned(self) -> str:
        """On-demand: "what did you learn today" — returns formatted string.

        Returns:
            Human-readable summary of today's lessons.
        """
        session_data = self._read_session_data()
        lessons = self._extract_patterns(session_data)

        if not lessons:
            return "Nothing new learned this session yet."

        lines = [f"I learned {len(lessons)} thing(s) this session:\n"]
        for i, lesson in enumerate(lessons, 1):
            lines.append(f"  {i}. [{lesson.pattern_type}] {lesson.lesson}")
            lines.append(f"     Evidence: {lesson.session_evidence[:120]}")
        return "\n".join(lines)

    # ── Step 1: Read session data ─────────────────────────────────────────────

    def _read_session_data(self) -> dict[str, Any]:
        """Load all interaction and tool event data for the current session.

        Returns:
            Dict with interactions, tool_events, session_id, and timing.
        """
        session_id = self.interaction_logger.session_id
        data: dict[str, Any] = {
            "session_id": session_id,
            "interactions": [],
            "tool_events": [],
            "tool_stats": [],
            "start_time": None,
            "end_time": None,
        }

        if not self.interaction_db.exists():
            return data

        try:
            conn = sqlite3.connect(str(self.interaction_db), timeout=10)
            conn.row_factory = sqlite3.Row

            # Interactions for this session
            rows = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            data["interactions"] = [dict(r) for r in rows]

            if data["interactions"]:
                data["start_time"] = data["interactions"][0].get("timestamp")
                data["end_time"] = data["interactions"][-1].get("timestamp")

            # Tool events for this session
            rows = conn.execute(
                "SELECT * FROM tool_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            data["tool_events"] = [dict(r) for r in rows]

            conn.close()
        except Exception as exc:
            logger.warning("Session Learner: failed to read interaction DB: %s", exc)

        # Tool stats from brain DB
        if self.brain_db_path.exists():
            try:
                conn = sqlite3.connect(str(self.brain_db_path), timeout=10)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT name, success_rate, total_runs, is_active, version "
                    "FROM tools"
                ).fetchall()
                data["tool_stats"] = [dict(r) for r in rows]
                conn.close()
            except Exception:
                pass

        return data

    # ── Step 2: Pattern extraction ────────────────────────────────────────────

    def _extract_patterns(self, session_data: dict[str, Any]) -> list[Lesson]:
        """Analyze session data and extract lessons.

        Args:
            session_data: Output of _read_session_data().

        Returns:
            List of Lesson instances found.
        """
        lessons: list[Lesson] = []

        lessons.extend(self._pattern_a_repeated_failures(session_data))
        lessons.extend(self._pattern_b_successful_tools(session_data))
        lessons.extend(self._pattern_c_misclassification(session_data))
        lessons.extend(self._pattern_d_agent_routing(session_data))

        return lessons

    def _pattern_a_repeated_failures(
        self, session_data: dict[str, Any]
    ) -> list[Lesson]:
        """Pattern A: Same intent failed 2+ times then succeeded.

        Args:
            session_data: Full session data dict.

        Returns:
            List of repeated-failure lessons.
        """
        lessons: list[Lesson] = []
        interactions = session_data.get("interactions", [])

        # Group by intent, track failures (error/empty response) vs successes
        intent_history: dict[str, list[dict]] = {}
        for ix in interactions:
            intent = ix.get("intent", "UNKNOWN")
            intent_history.setdefault(intent, []).append(ix)

        for intent, items in intent_history.items():
            failures: list[dict] = []
            first_success_after_failure: dict | None = None

            for item in items:
                response = item.get("response", "")
                is_failure = (
                    not response
                    or "error" in response.lower()[:100]
                    or "failed" in response.lower()[:100]
                    or "cannot" in response.lower()[:100]
                )
                if is_failure:
                    failures.append(item)
                elif failures:
                    # First success after a streak of failures
                    first_success_after_failure = item
                    break

            if len(failures) >= 2 and first_success_after_failure:
                failed_input = failures[0].get("user_input", "")[:80]
                success_input = first_success_after_failure.get("user_input", "")[:80]
                success_response = first_success_after_failure.get("response", "")[:80]
                lessons.append(Lesson(
                    pattern_type="A",
                    learned_at=datetime.now().isoformat(),
                    session_evidence=(
                        f"Intent {intent} failed {len(failures)}x "
                        f"(e.g. '{failed_input}'), then succeeded with "
                        f"'{success_input}'"
                    ),
                    lesson=(
                        f"When user asks [{intent}] and initial approach fails, "
                        f"use approach that produced: {success_response}"
                    ),
                    confidence=min(0.5 + len(failures) * 0.1, 0.9),
                ))

        return lessons

    def _pattern_b_successful_tools(
        self, session_data: dict[str, Any]
    ) -> list[Lesson]:
        """Pattern B: New tool created this session with good success rate.

        Args:
            session_data: Full session data dict.

        Returns:
            List of successful-tool lessons.
        """
        lessons: list[Lesson] = []
        tool_events = session_data.get("tool_events", [])
        tool_stats = session_data.get("tool_stats", [])

        # Find tools created this session
        created_tools: set[str] = set()
        for event in tool_events:
            if event.get("event_type") == "tool_created":
                created_tools.add(event.get("tool_name", ""))

        # Check their success rates
        stats_map = {t["name"]: t for t in tool_stats}
        for tool_name in created_tools:
            if not tool_name:
                continue
            stats = stats_map.get(tool_name)
            if not stats:
                continue
            rate = stats.get("success_rate", 0.0) or 0.0
            runs = stats.get("total_runs", 0) or 0
            if rate >= 0.7 and runs >= 1:
                # Find what triggered the creation
                trigger = ""
                for event in tool_events:
                    if (
                        event.get("event_type") == "tool_created"
                        and event.get("tool_name") == tool_name
                    ):
                        details = event.get("details", "")
                        if isinstance(details, str):
                            try:
                                details = json.loads(details)
                            except (json.JSONDecodeError, TypeError):
                                details = {}
                        trigger = (
                            details.get("user_request", "")[:80]
                            if isinstance(details, dict)
                            else ""
                        )
                        break

                lessons.append(Lesson(
                    pattern_type="B",
                    learned_at=datetime.now().isoformat(),
                    session_evidence=(
                        f"Tool '{tool_name}' created this session, "
                        f"success rate {rate:.0%} over {runs} runs"
                    ),
                    lesson=(
                        f"Tool '{tool_name}' handles '{trigger or 'its task type'}' "
                        f"reliably (rate: {rate:.0%})"
                    ),
                    confidence=min(0.5 + rate * 0.4, 0.95),
                ))

        return lessons

    def _pattern_c_misclassification(
        self, session_data: dict[str, Any]
    ) -> list[Lesson]:
        """Pattern C: Intent classified one way but corrected or wrong.

        Detects when a BRAIN_TASK/CODE_TASK gets a chat-like response
        or when CHAT gets tool-type handling.

        Args:
            session_data: Full session data dict.

        Returns:
            List of misclassification lessons.
        """
        lessons: list[Lesson] = []
        interactions = session_data.get("interactions", [])

        for ix in interactions:
            intent = ix.get("intent", "")
            user_input = ix.get("user_input", "")
            response = ix.get("response", "")

            # User said "no" / "that's wrong" / "not what I asked" in
            # the next interaction → the previous one was mishandled
            lower_resp = response.lower()[:150] if response else ""
            lower_input = user_input.lower()

            # Detect correction phrases in user input
            correction_phrases = (
                "no that's wrong", "not what i asked", "that's not right",
                "wrong", "no i meant", "i said", "try again",
            )
            if any(phrase in lower_input for phrase in correction_phrases):
                # Look backwards to find what was misclassified
                idx = interactions.index(ix)
                if idx > 0:
                    prev = interactions[idx - 1]
                    prev_intent = prev.get("intent", "")
                    prev_input = prev.get("user_input", "")[:80]
                    lessons.append(Lesson(
                        pattern_type="C",
                        learned_at=datetime.now().isoformat(),
                        session_evidence=(
                            f"User corrected after '{prev_input}' was classified "
                            f"as {prev_intent}"
                        ),
                        lesson=(
                            f"Phrase '{prev_input}' may be misclassified as "
                            f"{prev_intent} -- review intent classification"
                        ),
                        confidence=0.5,
                    ))

        return lessons

    def _pattern_d_agent_routing(
        self, session_data: dict[str, Any]
    ) -> list[Lesson]:
        """Pattern D: Task routed to wrong agent then re-routed.

        Detects when tool events show a fallback or re-route.

        Args:
            session_data: Full session data dict.

        Returns:
            List of agent-routing lessons.
        """
        lessons: list[Lesson] = []
        tool_events = session_data.get("tool_events", [])

        # Look for re-routing events in tool event details
        for event in tool_events:
            details_raw = event.get("details", "")
            if isinstance(details_raw, str):
                try:
                    details = json.loads(details_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                details = details_raw

            if not isinstance(details, dict):
                continue

            # Decision engine logs re-routes with "rerouted_from" field
            rerouted_from = details.get("rerouted_from")
            rerouted_to = details.get("rerouted_to") or details.get("agent")
            task_desc = details.get("task", details.get("user_request", ""))

            if rerouted_from and rerouted_to and rerouted_from != rerouted_to:
                lessons.append(Lesson(
                    pattern_type="D",
                    learned_at=datetime.now().isoformat(),
                    session_evidence=(
                        f"Task '{str(task_desc)[:80]}' routed from "
                        f"{rerouted_from} to {rerouted_to}"
                    ),
                    lesson=(
                        f"Task type '{str(task_desc)[:60]}' belongs to "
                        f"{rerouted_to}, not {rerouted_from}"
                    ),
                    confidence=0.6,
                ))

        return lessons

    # ── Step 3: Write lessons to personal model ──────────────────────────────

    def _write_lessons(self, lessons: list[Lesson]) -> int:
        """Persist lessons to personal_model.json.

        If the same lesson already exists, increment times_reinforced.
        If times_reinforced > 3, set confidence to 1.0.
        Conflicting lessons: keep the higher-confidence one.

        Args:
            lessons: List of Lesson instances to persist.

        Returns:
            Number of new or reinforced lessons.
        """
        if not lessons:
            return 0

        model = self.personal_model.model
        existing: list[dict] = model.setdefault("learned_patterns", [])

        new_count = 0

        for lesson in lessons:
            # Check for existing similar lesson (same pattern_type + similar text)
            matched = False
            for existing_lesson in existing:
                if (
                    existing_lesson.get("pattern_type") == lesson.pattern_type
                    and self._lessons_match(existing_lesson, lesson)
                ):
                    # Reinforce existing lesson
                    existing_lesson["times_reinforced"] = (
                        existing_lesson.get("times_reinforced", 1) + 1
                    )
                    if existing_lesson["times_reinforced"] > 3:
                        existing_lesson["confidence"] = 1.0
                    elif lesson.confidence > existing_lesson.get("confidence", 0):
                        existing_lesson["confidence"] = lesson.confidence
                    existing_lesson["learned_at"] = lesson.learned_at
                    matched = True
                    new_count += 1
                    break

            if not matched:
                # Check for conflicting lessons (same pattern_type, opposite advice)
                conflict_idx = self._find_conflict(existing, lesson)
                if conflict_idx is not None:
                    if lesson.confidence > existing[conflict_idx].get("confidence", 0):
                        existing[conflict_idx] = lesson.to_dict()
                        new_count += 1
                else:
                    existing.append(lesson.to_dict())
                    new_count += 1

        # Save
        self.personal_model._save()
        logger.info("Session Learner: wrote %d lessons to personal model", new_count)
        return new_count

    @staticmethod
    def _lessons_match(existing: dict, new: Lesson) -> bool:
        """Check if two lessons are about the same thing.

        Uses simple keyword overlap on the lesson text.

        Args:
            existing: Existing lesson dict from personal model.
            new: New Lesson instance.

        Returns:
            True if they refer to the same pattern.
        """
        old_words = set(existing.get("lesson", "").lower().split())
        new_words = set(new.lesson.lower().split())
        if not old_words or not new_words:
            return False
        overlap = len(old_words & new_words) / max(len(old_words), len(new_words))
        return overlap > 0.6

    @staticmethod
    def _find_conflict(existing: list[dict], new: Lesson) -> int | None:
        """Find an existing lesson that conflicts with the new one.

        Two lessons conflict if they have the same pattern_type and
        reference the same intent/tool but give opposite advice.

        Args:
            existing: List of existing lesson dicts.
            new: New Lesson instance.

        Returns:
            Index of conflicting lesson, or None.
        """
        for i, old in enumerate(existing):
            if old.get("pattern_type") != new.pattern_type:
                continue
            old_lesson = old.get("lesson", "").lower()
            new_lesson = new.lesson.lower()
            # Same subject but different recommendation
            old_words = set(old_lesson.split())
            new_words = set(new_lesson.split())
            subject_overlap = len(old_words & new_words) / max(
                len(old_words), len(new_words), 1
            )
            if subject_overlap > 0.4 and old_lesson != new_lesson:
                return i
        return None

    # ── Step 4: Build session summary ─────────────────────────────────────────

    def _build_summary(
        self, session_data: dict[str, Any], lessons_learned: int
    ) -> SessionSummary:
        """Build a SessionSummary from the raw session data.

        Args:
            session_data: Output of _read_session_data().
            lessons_learned: Number of lessons written this session.

        Returns:
            Populated SessionSummary.
        """
        interactions = session_data.get("interactions", [])
        tool_events = session_data.get("tool_events", [])

        # Duration
        duration = 0.0
        start_time = session_data.get("start_time")
        end_time = session_data.get("end_time")
        if start_time and end_time:
            try:
                dt_start = datetime.fromisoformat(start_time)
                dt_end = datetime.fromisoformat(end_time)
                duration = (dt_end - dt_start).total_seconds() / 60.0
            except (ValueError, TypeError):
                pass

        # Counts
        intent_counts: Counter[str] = Counter()
        for ix in interactions:
            intent_counts[ix.get("intent", "UNKNOWN")] += 1

        new_tools = sum(
            1 for e in tool_events if e.get("event_type") == "tool_created"
        )
        failed_tools = sum(
            1 for e in tool_events
            if e.get("event_type") in ("synthesis_failed", "tool_execution_failed")
        )

        # Top lesson
        model = self.personal_model.model
        patterns = model.get("learned_patterns", [])
        top_lesson = ""
        if patterns:
            # Most recently learned with highest confidence
            sorted_patterns = sorted(
                patterns,
                key=lambda p: (p.get("confidence", 0), p.get("learned_at", "")),
                reverse=True,
            )
            top_lesson = sorted_patterns[0].get("lesson", "")[:120]

        return SessionSummary(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            duration_minutes=round(duration, 1),
            tasks_completed=len(interactions),
            new_tools_created=new_tools,
            tools_failed=failed_tools,
            lessons_learned=lessons_learned,
            top_lesson=top_lesson,
            intent_breakdown=dict(intent_counts),
        )

    # ── Step 5: Save summary to file ──────────────────────────────────────────

    def _save_summary(self, summary: SessionSummary) -> None:
        """Write the session summary to data/session_summaries/.

        Args:
            summary: SessionSummary to persist.
        """
        filename = datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"
        path = self.summaries_dir / filename
        try:
            path.write_text(summary.format(), encoding="utf-8")
            logger.info("Session summary saved to %s", path)
        except Exception as exc:
            logger.error("Failed to save session summary: %s", exc)

    # ── Lesson loading for context injection ──────────────────────────────────

    @staticmethod
    def load_lessons_for_context(
        personal_model_path: Path,
        brain_db_path: Path | None = None,
    ) -> list[str]:
        """Load high-confidence lessons for context injection.

        Filters: confidence > 0.7, not referencing deactivated tools,
        returns max MAX_INJECTED_LESSONS sorted by confidence then recency.

        Args:
            personal_model_path: Path to personal_model.json.
            brain_db_path: Optional path to brain.db for active tool check.

        Returns:
            List of lesson strings ready for prompt injection.
        """
        if not personal_model_path.exists():
            return []

        try:
            model = json.loads(
                personal_model_path.read_text(encoding="utf-8")
            )
        except Exception:
            return []

        patterns: list[dict] = model.get("learned_patterns", [])
        if not patterns:
            return []

        # Get deactivated tool names
        deactivated: set[str] = set()
        if brain_db_path and brain_db_path.exists():
            try:
                conn = sqlite3.connect(str(brain_db_path), timeout=5)
                rows = conn.execute(
                    "SELECT name FROM tools WHERE is_active = 0"
                ).fetchall()
                deactivated = {r[0] for r in rows}
                conn.close()
            except Exception:
                pass

        # Filter and sort
        eligible: list[dict] = []
        for p in patterns:
            confidence = p.get("confidence", 0)
            if confidence < 0.7:
                continue
            lesson_text = p.get("lesson", "")
            # Skip if it references a deactivated tool
            if deactivated and any(
                tool_name in lesson_text for tool_name in deactivated
            ):
                continue
            eligible.append(p)

        # Sort by confidence desc, then learned_at desc
        eligible.sort(
            key=lambda p: (p.get("confidence", 0), p.get("learned_at", "")),
            reverse=True,
        )

        return [
            p["lesson"]
            for p in eligible[:MAX_INJECTED_LESSONS]
            if p.get("lesson")
        ]
