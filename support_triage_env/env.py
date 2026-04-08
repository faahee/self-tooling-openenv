from __future__ import annotations

import copy
from typing import Any

from .models import (
    SupportTriageAction,
    SupportTriageObservation,
    SupportTriageState,
    SupportTriageStepResult,
)
from .tasks import TASKS, describe_action, grade_task


class SupportTriageEnvironment:
    def __init__(self, default_task: str = "easy_refund_triage") -> None:
        self.default_task = default_task
        self._task_name = default_task
        self._tickets = []
        self._step_count = 0
        self._done = False
        self._last_action_error: str | None = None
        self._score = 0.0
        self._history: list[SupportTriageAction] = []
        self.reset(default_task)

    @property
    def task(self):
        return TASKS[self._task_name]

    def _allowed_actions(self) -> list[str]:
        return ["classify", "prioritize", "assign", "reply", "set_status", "noop"]

    def _ticket_map(self):
        return {ticket.ticket_id: ticket for ticket in self._tickets}

    def _build_observation(self) -> SupportTriageObservation:
        return SupportTriageObservation(
            task_name=self._task_name,
            objective=self.task.objective,
            step_count=self._step_count,
            max_steps=self.task.max_steps,
            cumulative_score=self._score,
            last_action_error=self._last_action_error,
            tickets=self._tickets,
            allowed_actions=self._allowed_actions(),
        )

    def reset(self, task_name: str | None = None) -> SupportTriageStepResult:
        self._task_name = task_name or self.default_task
        self._tickets = [copy.deepcopy(ticket) for ticket in TASKS[self._task_name].tickets]
        self._step_count = 0
        self._done = False
        self._last_action_error = None
        self._score = 0.0
        self._history = []
        return SupportTriageStepResult(
            observation=self._build_observation(),
            reward=0.0,
            done=False,
            info={"score": 0.0, "grader": {}},
        )

    def state(self) -> SupportTriageState:
        return SupportTriageState(
            task_name=self._task_name,
            objective=self.task.objective,
            step_count=self._step_count,
            max_steps=self.task.max_steps,
            done=self._done,
            cumulative_score=self._score,
            last_action_error=self._last_action_error,
            tickets=self._tickets,
            action_history=self._history,
        )

    def _apply_action(self, action: SupportTriageAction) -> float:
        penalty = 0.0
        if action.action_type == "noop":
            return 0.03
        if not action.ticket_id:
            self._last_action_error = "ticket_id is required"
            return 0.10
        ticket = self._ticket_map().get(action.ticket_id)
        if not ticket:
            self._last_action_error = f"unknown ticket_id: {action.ticket_id}"
            return 0.10

        signature = (action.action_type, action.ticket_id, action.category, action.priority, action.queue, action.status, action.message)
        recent = [
            (item.action_type, item.ticket_id, item.category, item.priority, item.queue, item.status, item.message)
            for item in self._history[-3:]
        ]
        if signature in recent:
            penalty += 0.03

        self._last_action_error = None
        if action.action_type == "classify":
            if not action.category:
                self._last_action_error = "category is required"
                return 0.10 + penalty
            ticket.category = action.category
        elif action.action_type == "prioritize":
            if not action.priority:
                self._last_action_error = "priority is required"
                return 0.10 + penalty
            ticket.priority = action.priority
        elif action.action_type == "assign":
            if not action.queue:
                self._last_action_error = "queue is required"
                return 0.10 + penalty
            ticket.queue = action.queue
        elif action.action_type == "reply":
            if not action.message:
                self._last_action_error = "message is required"
                return 0.10 + penalty
            ticket.latest_reply = action.message
        elif action.action_type == "set_status":
            if not action.status:
                self._last_action_error = "status is required"
                return 0.10 + penalty
            ticket.status = action.status
        else:
            self._last_action_error = f"unsupported action_type: {action.action_type}"
            return 0.10 + penalty
        return penalty

    def step(self, action: SupportTriageAction) -> SupportTriageStepResult:
        if self._done:
            return SupportTriageStepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"score": self._score, "grader": {}, "last_action_error": "episode already done"},
            )

        previous_score = self._score
        self._step_count += 1
        penalty = self._apply_action(action)
        self._history.append(action)
        current_score, components = grade_task(self._task_name, self._tickets)
        self._score = current_score
        improvement = max(0.0, current_score - previous_score)
        reward_value = max(0.0, min(1.0, improvement + 0.05 - penalty))
        self._done = current_score >= 0.999 or self._step_count >= self.task.max_steps
        info: dict[str, Any] = {
            "score": current_score,
            "grader": components,
            "penalty": round(penalty, 3),
            "action_signature": describe_action(action),
            "last_action_error": self._last_action_error,
        }
        return SupportTriageStepResult(
            observation=self._build_observation(),
            reward=round(reward_value, 3),
            done=self._done,
            info=info,
        )
