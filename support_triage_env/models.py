from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal["classify", "prioritize", "assign", "reply", "set_status", "noop"]
TicketCategory = Literal["billing", "technical", "security", "product", "spam", "general"]
TicketPriority = Literal["low", "normal", "high", "urgent"]
TicketStatus = Literal["open", "pending", "resolved", "closed", "spam"]


class TicketRecord(BaseModel):
    ticket_id: str
    sender: str
    subject: str
    body: str
    customer_tier: str
    category: TicketCategory | None = None
    priority: TicketPriority | None = None
    queue: str | None = None
    status: TicketStatus = "open"
    latest_reply: str | None = None
    notes: list[str] = Field(default_factory=list)


class SupportTriageObservation(BaseModel):
    task_name: str
    objective: str
    step_count: int
    max_steps: int
    cumulative_score: float
    last_action_error: str | None = None
    tickets: list[TicketRecord]
    allowed_actions: list[str]


class SupportTriageAction(BaseModel):
    action_type: ActionType
    ticket_id: str | None = None
    category: TicketCategory | None = None
    priority: TicketPriority | None = None
    queue: str | None = None
    status: TicketStatus | None = None
    message: str | None = None


class SupportTriageReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    penalty: float = Field(default=0.0, ge=0.0, le=1.0)


class SupportTriageState(BaseModel):
    task_name: str
    objective: str
    step_count: int
    max_steps: int
    done: bool
    cumulative_score: float
    last_action_error: str | None = None
    tickets: list[TicketRecord]
    action_history: list[SupportTriageAction] = Field(default_factory=list)


class SupportTriageStepResult(BaseModel):
    observation: SupportTriageObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_name: str | None = None
