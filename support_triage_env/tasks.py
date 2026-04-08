from __future__ import annotations

from dataclasses import dataclass

from .models import SupportTriageAction, TicketRecord


@dataclass(frozen=True)
class TicketTarget:
    category: str
    priority: str
    queue: str
    status: str
    reply_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    difficulty: str
    objective: str
    max_steps: int
    tickets: tuple[TicketRecord, ...]
    targets: dict[str, TicketTarget]

    def grade(self, tickets: list[TicketRecord]) -> tuple[float, dict[str, float]]:
        ticket_map = {ticket.ticket_id: ticket for ticket in tickets}
        components: dict[str, float] = {}
        total = 0.0
        for ticket_id, target in self.targets.items():
            ticket = ticket_map[ticket_id]
            ticket_score = 0.0
            if ticket.category == target.category:
                ticket_score += 0.20
            if ticket.priority == target.priority:
                ticket_score += 0.15
            if (ticket.queue or "").lower() == target.queue.lower():
                ticket_score += 0.20
            if ticket.status == target.status:
                ticket_score += 0.15
            if target.reply_keywords:
                reply = (ticket.latest_reply or "").lower()
                hits = sum(1 for keyword in target.reply_keywords if keyword.lower() in reply)
                ticket_score += 0.30 * (hits / len(target.reply_keywords))
            else:
                ticket_score += 0.30
            components[ticket_id] = round(ticket_score, 3)
            total += ticket_score
        return round(total / max(len(self.targets), 1), 3), components


TASKS: dict[str, TaskDefinition] = {
    "easy_refund_triage": TaskDefinition(
        name="easy_refund_triage",
        difficulty="easy",
        objective="Triage a single billing refund request. Classify it, set a sensible priority, assign the right queue, and send a reassuring reply.",
        max_steps=6,
        tickets=(
            TicketRecord(
                ticket_id="T1",
                sender="maya@acme.co",
                subject="Charged twice after plan upgrade",
                body="I upgraded to Pro this morning and my card was charged twice. Please help me get the duplicate payment refunded.",
                customer_tier="pro",
            ),
        ),
        targets={
            "T1": TicketTarget(
                category="billing",
                priority="normal",
                queue="billing",
                status="pending",
                reply_keywords=("refund", "billing", "24 hours"),
            )
        },
    ),
    "medium_inbox_triage": TaskDefinition(
        name="medium_inbox_triage",
        difficulty="medium",
        objective="Triage three incoming support tickets: an outage, a refund request, and obvious spam.",
        max_steps=10,
        tickets=(
            TicketRecord(
                ticket_id="T1",
                sender="ops@northstar.io",
                subject="Production login outage",
                body="Our whole success team cannot log in. This started 10 minutes ago and we are blocked with customers waiting.",
                customer_tier="enterprise",
            ),
            TicketRecord(
                ticket_id="T2",
                sender="leo@example.com",
                subject="Need refund for duplicate invoice",
                body="I accidentally paid invoice INV-221 twice. Can you refund the duplicate charge?",
                customer_tier="starter",
            ),
            TicketRecord(
                ticket_id="T3",
                sender="seo-growth-fast.biz",
                subject="Double your traffic instantly",
                body="Guaranteed backlinks and crypto returns. Reply now for premium placement.",
                customer_tier="unknown",
            ),
        ),
        targets={
            "T1": TicketTarget(
                category="technical",
                priority="urgent",
                queue="engineering",
                status="pending",
                reply_keywords=("incident", "engineers", "updates"),
            ),
            "T2": TicketTarget(
                category="billing",
                priority="normal",
                queue="billing",
                status="pending",
                reply_keywords=("refund", "invoice", "review"),
            ),
            "T3": TicketTarget(
                category="spam",
                priority="low",
                queue="abuse",
                status="spam",
            ),
        },
    ),
    "hard_enterprise_inbox": TaskDefinition(
        name="hard_enterprise_inbox",
        difficulty="hard",
        objective="Handle an enterprise inbox involving a production outage, a privacy request, a billing dispute, and a feature request.",
        max_steps=14,
        tickets=(
            TicketRecord(
                ticket_id="T1",
                sender="cfo@globex.com",
                subject="URGENT: dashboards unavailable for leadership",
                body="Executive dashboards are down across our EU region and finance cannot close the quarter. We need an incident response now.",
                customer_tier="enterprise",
            ),
            TicketRecord(
                ticket_id="T2",
                sender="privacy@globex.com",
                subject="Data deletion request for departed employee",
                body="Please remove personal data related to former employee Marta Ionescu. We need confirmation that the request is being processed.",
                customer_tier="enterprise",
            ),
            TicketRecord(
                ticket_id="T3",
                sender="owner@studiofizz.co",
                subject="Downgrade dispute and invoice complaint",
                body="We downgraded last week but were still invoiced for the higher plan. I want the billing reviewed and any overcharge corrected.",
                customer_tier="business",
            ),
            TicketRecord(
                ticket_id="T4",
                sender="pm@sunlit.app",
                subject="Feature request: shared inbox analytics",
                body="Our team would love analytics for response-time trends across agents. Is this on your roadmap?",
                customer_tier="pro",
            ),
        ),
        targets={
            "T1": TicketTarget(
                category="technical",
                priority="urgent",
                queue="engineering",
                status="pending",
                reply_keywords=("incident", "priority", "updates"),
            ),
            "T2": TicketTarget(
                category="security",
                priority="high",
                queue="security",
                status="pending",
                reply_keywords=("privacy", "verify", "processing"),
            ),
            "T3": TicketTarget(
                category="billing",
                priority="normal",
                queue="billing",
                status="pending",
                reply_keywords=("billing", "refund", "review"),
            ),
            "T4": TicketTarget(
                category="product",
                priority="low",
                queue="product",
                status="pending",
                reply_keywords=("feature", "product", "roadmap"),
            ),
        },
    ),
}


def list_tasks() -> list[str]:
    return list(TASKS.keys())


def grade_task(task_name: str, tickets: list[TicketRecord]) -> tuple[float, dict[str, float]]:
    return TASKS[task_name].grade(tickets)


def describe_action(action: SupportTriageAction) -> str:
    target = action.ticket_id or "none"
    return f"{action.action_type}:{target}"
