from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - local fallback when SDK is unavailable
    OpenAI = Any

from support_triage_env import SupportTriageAction, SupportTriageEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("SUPPORT_TRIAGE_TASK", "easy_refund_triage")
BENCHMARK = os.getenv("SUPPORT_TRIAGE_BENCHMARK", "support-triage-openenv")
TEMPERATURE = 0.1
MAX_TOKENS = 250


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = """
You are solving a support inbox triage task.
Return exactly one JSON object with keys:
action_type, ticket_id, category, priority, queue, status, message
Only include values needed for the chosen action.
Prefer focused support operations such as classify, prioritize, assign, reply, or set_status.
""".strip()


def heuristic_action(observation) -> SupportTriageAction:
    def ticket_complete(ticket) -> bool:
        if ticket.category == "spam":
            return ticket.status == "spam"
        return bool(
            ticket.category
            and ticket.priority
            and ticket.queue
            and ticket.latest_reply
            and ticket.status == "pending"
        )

    for ticket in observation.tickets:
        if ticket_complete(ticket):
            continue
        body = f"{ticket.subject} {ticket.body}".lower()
        if ticket.category is None:
            if any(w in body for w in ("refund", "invoice", "charge", "billing")):
                return SupportTriageAction(action_type="classify", ticket_id=ticket.ticket_id, category="billing")
            if any(w in body for w in ("outage", "down", "login", "incident")):
                return SupportTriageAction(action_type="classify", ticket_id=ticket.ticket_id, category="technical")
            if any(w in body for w in ("privacy", "delete", "security", "personal data")):
                return SupportTriageAction(action_type="classify", ticket_id=ticket.ticket_id, category="security")
            if any(w in body for w in ("feature", "roadmap", "request")):
                return SupportTriageAction(action_type="classify", ticket_id=ticket.ticket_id, category="product")
            if any(w in body for w in ("backlinks", "crypto", "traffic instantly")):
                return SupportTriageAction(action_type="classify", ticket_id=ticket.ticket_id, category="spam")
            return SupportTriageAction(action_type="classify", ticket_id=ticket.ticket_id, category="general")
        if ticket.priority is None:
            if "outage" in body or "urgent" in body or "down" in body:
                return SupportTriageAction(action_type="prioritize", ticket_id=ticket.ticket_id, priority="urgent")
            if "privacy" in body or "security" in body:
                return SupportTriageAction(action_type="prioritize", ticket_id=ticket.ticket_id, priority="high")
            if ticket.category == "product":
                return SupportTriageAction(action_type="prioritize", ticket_id=ticket.ticket_id, priority="low")
            return SupportTriageAction(action_type="prioritize", ticket_id=ticket.ticket_id, priority="normal")
        if ticket.queue is None:
            queue = {
                "billing": "billing",
                "technical": "engineering",
                "security": "security",
                "product": "product",
                "spam": "abuse",
                "general": "support",
            }[ticket.category]
            return SupportTriageAction(action_type="assign", ticket_id=ticket.ticket_id, queue=queue)
        if ticket.latest_reply is None and ticket.category != "spam":
            message = {
                "billing": "Thanks for flagging this. Our billing team will review the invoice and refund request within 24 hours.",
                "technical": "We have opened an incident and our engineers are investigating. We will keep you updated with priority status changes.",
                "security": "We are processing this privacy request. Our security team will verify the details and follow up on processing.",
                "product": "Thank you for the feature request. I have shared this with our product team for roadmap review.",
                "general": "Thanks for reaching out. Our support team is reviewing your request.",
            }.get(ticket.category, "Thanks for reaching out. Our team is reviewing this.")
            return SupportTriageAction(action_type="reply", ticket_id=ticket.ticket_id, message=message)
        if ticket.status == "open":
            status = "spam" if ticket.category == "spam" else "pending"
            return SupportTriageAction(action_type="set_status", ticket_id=ticket.ticket_id, status=status)
    return SupportTriageAction(action_type="noop", message="done")


def get_model_action(client: Optional[OpenAI], observation) -> SupportTriageAction:
    payload = {
        "objective": observation.objective,
        "step_count": observation.step_count,
        "max_steps": observation.max_steps,
        "cumulative_score": observation.cumulative_score,
        "last_action_error": observation.last_action_error,
        "tickets": [ticket.model_dump() for ticket in observation.tickets],
    }
    if client is None:
        return heuristic_action(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            return SupportTriageAction.model_validate_json(raw[start:end + 1])
    except Exception:
        pass
    return heuristic_action(observation)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    try:
        result = await env.reset(TASK_NAME)
        while not result.done:
            observation = result.observation
            action = get_model_action(client, observation)
            result = await env.step(action)
            rewards.append(result.reward)
            steps_taken += 1
            action_str = json.dumps(action.model_dump(exclude_none=True), ensure_ascii=True, separators=(",", ":"))
            log_step(
                step=steps_taken,
                action=action_str,
                reward=result.reward,
                done=result.done,
                error=result.info.get("last_action_error"),
            )
            if result.done:
                break
        score = float(result.info.get("score", 0.0))
        success = score >= 0.7
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
