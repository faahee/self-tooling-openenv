"""Personal Model — A living, auto-updating model of the user."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PersonalModel:
    """Tracks 5 dimensions of who the user is and updates them after every interaction.

    Thread-safe: all writes to the model use a lock.
    """

    DEFAULT_MODEL: dict = {
        "name": "Jarvis",
        "goals": {
            "long_term": [
                "Build transformative deep tech — AI OS agent, autonomous systems",
                "Build VALLUGE fashion AI startup for Indian market",
                "Master ML/AI from fundamentals to deployment",
            ],
            "current_projects": ["JARVIS OS agent", "VALLUGE", "ML study plan"],
            "short_term": [],
        },
        "cognitive_patterns": {
            "peak_hours": [],
            "avg_session_length_mins": 0,
            "preferred_detail_level": "detailed",
            "learning_style": "examples",
        },
        "preferences": {
            "communication_style": "direct",
            "tts_enabled": True,
            "proactive_suggestions": True,
            "code_language": "python",
        },
        "interaction_patterns": {
            "total_interactions": 0,
            "most_used_intents": {},
            "common_topics": [],
            "last_active": None,
        },
        "emotional_patterns": {
            "stress_indicators": [],
            "energized_indicators": [],
        },
    }

    def __init__(self, config: dict) -> None:
        """Initialize the personal model.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.path = Path(config["memory"]["personal_model_path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.model: dict = self._load()

    def get_context_string(self) -> str:
        """Return a formatted string for injection into the system prompt.

        Returns:
            Multi-line string summarising the user's profile.
        """
        m = self.model
        goals = m.get("goals", {})
        prefs = m.get("preferences", {})
        patterns = m.get("cognitive_patterns", {})

        long_term = ", ".join(goals.get("long_term", [])[:3]) or "None set"
        projects = ", ".join(goals.get("current_projects", [])[:3]) or "None"
        style = prefs.get("communication_style", "direct")
        detail = patterns.get("preferred_detail_level", "detailed")

        return (
            f"Name: {m.get('name', 'User')}\n"
            f"Long-term goals: {long_term}\n"
            f"Current projects: {projects}\n"
            f"Communication style: {style}\n"
            f"Preferred detail level: {detail}"
        )

    def update_from_interaction(
        self, user_input: str, response: str, intent: str
    ) -> None:
        """Update the personal model after an interaction.

        Args:
            user_input: The user's input text.
            response: JARVIS response text.
            intent: Classified intent label.
        """
        with self._lock:
            ip = self.model.setdefault("interaction_patterns", {})
            ip["total_interactions"] = ip.get("total_interactions", 0) + 1
            ip["last_active"] = datetime.now().isoformat()

            intents = ip.setdefault("most_used_intents", {})
            intents[intent] = intents.get(intent, 0) + 1

            # Update peak hours
            hour = datetime.now().hour
            cp = self.model.setdefault("cognitive_patterns", {})
            peak = cp.setdefault("peak_hours", [])
            peak.append(hour)
            # Keep only last 200 data points
            if len(peak) > 200:
                cp["peak_hours"] = peak[-200:]

            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load from JSON or return a copy of DEFAULT_MODEL."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return data
            except Exception as exc:
                logger.warning("Could not load personal model, using defaults: %s", exc)
        return json.loads(json.dumps(self.DEFAULT_MODEL))  # deep copy

    def _save(self) -> None:
        """Write the model to disk (called under lock)."""
        try:
            self.path.write_text(
                json.dumps(self.model, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Failed to save personal model: %s", exc)
