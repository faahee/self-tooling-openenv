"""Scheduler Agent — Schedule tasks, set reminders, run workflows on a timer."""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class SchedulerAgent:
    """Manages scheduled tasks using APScheduler.

    Supports:
      - One-time tasks at a specific time ("remind me at 3pm to call John")
      - Interval tasks ("check email every 30 minutes")
      - Cron tasks ("run backup every day at midnight")
    """

    def __init__(self, config: dict, orchestrator_callback: Callable, ui_layer) -> None:
        """
        Args:
            config: JARVIS config dict.
            orchestrator_callback: Async callable (handle_input) to run scheduled tasks.
            ui_layer: UILayer for notifications.
        """
        self.config = config
        self._callback = orchestrator_callback
        self.ui = ui_layer
        self._scheduler = AsyncIOScheduler()
        self._jobs: dict[str, dict] = {}  # job_id -> {description, command, next_run}
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._scheduler.start()
            self._started = True
            logger.info("SchedulerAgent started.")

    def stop(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle(self, user_input: str, context: dict) -> str:
        """Route scheduling commands.

        Args:
            user_input: Natural-language scheduling request.
            context: Context dict.

        Returns:
            Confirmation or list string.
        """
        low = user_input.lower()

        if any(w in low for w in ("list", "show", "what tasks", "scheduled tasks", "what's scheduled")):
            return self._list_jobs()

        if any(w in low for w in ("cancel", "remove", "delete", "stop task", "unschedule")):
            return self._cancel_job(user_input)

        if any(w in low for w in ("remind", "reminder", "alert")):
            return await self._schedule_reminder(user_input)

        if any(w in low for w in ("every", "interval", "repeat", "recurring", "each")):
            return await self._schedule_interval(user_input)

        if any(w in low for w in ("at ", "on ", "daily", "weekly", "midnight", "noon", "morning", "evening")):
            return await self._schedule_cron(user_input)

        return await self._schedule_auto(user_input)

    # ── Scheduling ────────────────────────────────────────────────────────────

    async def _schedule_reminder(self, user_input: str) -> str:
        """Schedule a one-time reminder."""
        # Parse time: "at 3pm", "in 30 minutes", "at 15:30"
        run_at = self._parse_time(user_input)
        if run_at is None:
            return "Could not parse time from your request. Try: 'remind me at 3pm to call John'"

        # Extract the action/message
        msg_match = re.search(r'(?:to|that|about|:)\s+(.+)', user_input, re.I)
        message = msg_match.group(1).strip() if msg_match else user_input

        job_id = f"reminder_{len(self._jobs)+1}"
        self._scheduler.add_job(
            self._notify,
            trigger=DateTrigger(run_date=run_at),
            args=[message, job_id],
            id=job_id,
        )
        self._jobs[job_id] = {"description": f"Reminder: {message}", "time": run_at.strftime("%Y-%m-%d %H:%M")}
        return f"Reminder set for {run_at.strftime('%I:%M %p on %b %d')} — '{message}'"

    async def _schedule_interval(self, user_input: str) -> str:
        """Schedule a repeating interval task."""
        # Parse interval: "every 30 minutes", "every 2 hours", "every day"
        n = 1
        unit = "hours"
        m = re.search(r'every\s+(\d+)?\s*(second|minute|hour|day|week)', user_input, re.I)
        if m:
            n = int(m.group(1)) if m.group(1) else 1
            unit = m.group(2).lower()
            if not unit.endswith("s"):
                unit += "s"

        # Extract the command to run
        cmd_match = re.search(r'(?:run|execute|do|check|send)\s+(.+?)(?:\s+every|\s*$)', user_input, re.I)
        command = cmd_match.group(1).strip() if cmd_match else user_input

        kwargs = {unit: n}
        job_id = f"interval_{len(self._jobs)+1}"
        self._scheduler.add_job(
            self._run_command,
            trigger=IntervalTrigger(**kwargs),
            args=[command, job_id],
            id=job_id,
        )
        self._jobs[job_id] = {"description": command, "interval": f"every {n} {unit}"}
        return f"Scheduled '{command}' to run every {n} {unit}. Job ID: {job_id}"

    async def _schedule_cron(self, user_input: str) -> str:
        """Schedule a cron-style task."""
        low = user_input.lower()
        hour, minute = 9, 0  # default: 9am

        if "midnight" in low:  hour, minute = 0, 0
        elif "noon" in low:    hour, minute = 12, 0
        elif "morning" in low: hour, minute = 8, 0
        elif "evening" in low: hour, minute = 18, 0
        else:
            t = re.search(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', user_input, re.I)
            if t:
                hour   = int(t.group(1))
                minute = int(t.group(2)) if t.group(2) else 0
                if t.group(3) and t.group(3).lower() == "pm" and hour < 12:
                    hour += 12

        day_of_week = None
        if "monday" in low:    day_of_week = "mon"
        elif "tuesday" in low: day_of_week = "tue"
        elif "wednesday" in low: day_of_week = "wed"
        elif "thursday" in low: day_of_week = "thu"
        elif "friday" in low:  day_of_week = "fri"
        elif "saturday" in low: day_of_week = "sat"
        elif "sunday" in low:  day_of_week = "sun"
        elif "weekday" in low: day_of_week = "mon-fri"
        elif "weekend" in low: day_of_week = "sat,sun"

        cmd_match = re.search(r'(?:run|execute|do|check|send|backup)\s+(.+?)(?:\s+(?:every|at|on)\b|\s*$)', user_input, re.I)
        command = cmd_match.group(1).strip() if cmd_match else user_input

        job_id = f"cron_{len(self._jobs)+1}"
        trigger = CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week)
        self._scheduler.add_job(
            self._run_command,
            trigger=trigger,
            args=[command, job_id],
            id=job_id,
        )
        time_str = f"{hour:02d}:{minute:02d}" + (f" on {day_of_week}" if day_of_week else " daily")
        self._jobs[job_id] = {"description": command, "schedule": time_str}
        return f"Scheduled '{command}' to run at {time_str}. Job ID: {job_id}"

    async def _schedule_auto(self, user_input: str) -> str:
        """Auto-detect scheduling type."""
        low = user_input.lower()
        if "every" in low:
            return await self._schedule_interval(user_input)
        if any(w in low for w in ("at ", "daily", "weekly")):
            return await self._schedule_cron(user_input)
        return await self._schedule_reminder(user_input)

    # ── Job management ────────────────────────────────────────────────────────

    def _list_jobs(self) -> str:
        if not self._jobs:
            return "No scheduled tasks."
        lines = ["Scheduled tasks:"]
        for jid, info in self._jobs.items():
            desc = info.get("description", "?")
            when = info.get("time") or info.get("interval") or info.get("schedule", "?")
            lines.append(f"  [{jid}] {desc} — {when}")
        return "\n".join(lines)

    def _cancel_job(self, user_input: str) -> str:
        id_match = re.search(r'\b((?:reminder|interval|cron)_\d+|\d+)\b', user_input)
        if id_match:
            job_id = id_match.group(1)
            if job_id.isdigit():
                # Find by number
                ids = list(self._jobs.keys())
                idx = int(job_id) - 1
                job_id = ids[idx] if 0 <= idx < len(ids) else ""
            if job_id in self._jobs:
                try:
                    self._scheduler.remove_job(job_id)
                except Exception:
                    pass
                desc = self._jobs.pop(job_id)["description"]
                return f"Cancelled task: '{desc}'"
        return "Could not find the task. Use 'list tasks' to see job IDs."

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _notify(self, message: str, job_id: str) -> None:
        """Fire a notification reminder."""
        self.ui.console.print(f"\n  [bold yellow]** REMINDER: {message}[/bold yellow]")
        try:
            self.ui.speak(f"Reminder: {message}")
        except Exception:
            pass
        # Remove one-time jobs from tracking
        self._jobs.pop(job_id, None)

    async def _run_command(self, command: str, job_id: str) -> None:
        """Execute a scheduled command through the orchestrator."""
        self.ui.console.print(f"\n  [bold cyan]>> Scheduled task running: {command}[/bold cyan]")
        try:
            result = await self._callback(command)
            logger.info("Scheduled task [%s] result: %s", job_id, result[:100])
        except Exception as e:
            logger.error("Scheduled task [%s] failed: %s", job_id, e)

    # ── Time parsing ──────────────────────────────────────────────────────────

    def _parse_time(self, text: str) -> datetime | None:
        now = datetime.now()
        low = text.lower()

        # "in X minutes/hours"
        m = re.search(r'in\s+(\d+)\s*(minute|hour|second|day)', low)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            delta = {
                "second": timedelta(seconds=n),
                "minute": timedelta(minutes=n),
                "hour":   timedelta(hours=n),
                "day":    timedelta(days=n),
            }.get(unit, timedelta(minutes=n))
            return now + delta

        # "at HH:MM" or "at 3pm"
        m = re.search(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', low)
        if m:
            hour   = int(m.group(1))
            minute = int(m.group(2)) if m.group(2) else 0
            if m.group(3) == "pm" and hour < 12:
                hour += 12
            elif m.group(3) == "am" and hour == 12:
                hour = 0
            run_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if run_at <= now:
                run_at += timedelta(days=1)
            return run_at

        # "tomorrow at..."
        if "tomorrow" in low:
            m2 = re.search(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', low)
            if m2:
                hour   = int(m2.group(1))
                minute = int(m2.group(2)) if m2.group(2) else 0
                if m2.group(3) == "pm" and hour < 12:
                    hour += 12
                return (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)

        return None
