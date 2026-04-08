---
title: self-tooling-openenv
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - agent-eval
  - self-tooling
  - customer-support
  - docker
---

# self-tooling-openenv

`self-tooling-openenv` is an OpenEnv-ready benchmark and reference project for evaluating agents on real support triage work, with the broader JARVIS codebase included as the original self-tooling system that can reason about missing capabilities, synthesize new tools, review them, and sandbox them before use.

## What This Repo Contains

This repository has two closely related layers:

1. The OpenEnv submission package
   - A complete real-world environment for customer support inbox triage.
   - Implements typed `Observation`, `Action`, and `Reward` models.
   - Exposes `reset()`, `step()`, and `state()` through an API and local client.
   - Includes deterministic tasks, graders, Docker packaging, and a required root `inference.py`.

2. The original JARVIS self-tooling architecture
   - A larger Windows agent system with agents, routing, tool search, MCP integration, and a tool synthesizer.
   - This is the part that gives the repo its `self-tooling-openenv` identity.
   - The synthesizer is important because it shows how an agent can decide when existing tools are enough and when a new tool should be generated, validated, and safely executed.

## Why "Self-Tooling"

Most agent benchmarks test tool use. This project is built around a stronger idea: an agent should also be able to reason about missing capabilities and create new tools when the task requires them.

In the JARVIS codebase, that idea lives in:

- [brain/decision_engine.py](/E:/projects/jarvis/brain/decision_engine.py)
- [brain/tool_synthesizer.py](/E:/projects/jarvis/brain/tool_synthesizer.py)

The OpenEnv benchmark in this repo focuses on structured support triage, while the surrounding JARVIS system demonstrates the larger self-tooling architecture that inspired it.

## Tool Synthesizer

The tool synthesizer is the clearest expression of the repo's core idea.

### What it does

When the agent sees a task that existing agents or tools cannot solve well, it can:

- analyze the request,
- describe the exact capability needed,
- plan a concrete Python tool,
- generate the implementation,
- review the implementation,
- run it in a sandbox,
- and register it for reuse.

This is not just tool calling. It is tool creation with safeguards.

### Main components

- [brain/decision_engine.py](/E:/projects/jarvis/brain/decision_engine.py)
  - Understands intent.
  - Chooses between existing agents, existing tools, MCP tools, or creating a new tool.
  - Uses explicit rules for cases that should never trigger synthesis, such as simple OS actions.

- [brain/tool_synthesizer.py](/E:/projects/jarvis/brain/tool_synthesizer.py)
  - Runs the structured synthesis pipeline.
  - Converts a task request into a `ToolBlueprint`.
  - Generates Python code from the blueprint.
  - Reviews and validates the generated code.
  - Executes candidate code in a sandbox before registration.

### The 5-phase synthesis pipeline

The synthesizer in [brain/tool_synthesizer.py](/E:/projects/jarvis/brain/tool_synthesizer.py) follows a strict 5-phase pipeline:

1. Deep reasoning
   - Produces a structured `ToolBlueprint`.
   - Captures task summary, parameters, imports, approach, pseudocode, edge cases, Windows constraints, risk level, and confidence.

2. Blueprint validation
   - Checks whether the proposed tool specification is complete enough before any code is written.

3. Code generation
   - Translates the blueprint into a small, self-contained Python function.
   - Keeps imports inside the function body and requires dictionary-based success/failure returns.

4. Self review
   - Runs a review prompt over the generated code.
   - Checks imports, return structure, error handling, type safety, and whether the code really matches the blueprint.

5. Sandbox and quality checks
   - Executes the candidate implementation in a constrained environment.
   - Rejects bad or unsafe results before the tool is added to the registry.

### Why this matters

This part of the repo is useful beyond the benchmark itself:

- It demonstrates how an agent can expand its own capabilities.
- It separates planning, generation, validation, and execution.
- It reduces unsafe "just run generated code" behavior.
- It creates a reusable tool inventory over time instead of solving each task from scratch.

## OpenEnv Environment

The submission environment models a real support operations workflow. An agent must triage customer tickets by taking structured actions against a live environment state.

Agents can:

- classify tickets,
- set priority,
- assign tickets to the correct queue,
- write a customer-facing reply,
- and update status.

Core implementation files:

- [support_triage_env/models.py](/E:/projects/jarvis/support_triage_env/models.py)
- [support_triage_env/tasks.py](/E:/projects/jarvis/support_triage_env/tasks.py)
- [support_triage_env/env.py](/E:/projects/jarvis/support_triage_env/env.py)
- [support_triage_env/server.py](/E:/projects/jarvis/support_triage_env/server.py)
- [support_triage_env/client.py](/E:/projects/jarvis/support_triage_env/client.py)
- [openenv.yaml](/E:/projects/jarvis/openenv.yaml)
- [inference.py](/E:/projects/jarvis/inference.py)
- [Dockerfile](/E:/projects/jarvis/Dockerfile)

## Environment API

### Observation space

Defined in [support_triage_env/models.py](/E:/projects/jarvis/support_triage_env/models.py) as `SupportTriageObservation`.

Fields include:

- `task_name`
- `objective`
- `step_count`
- `max_steps`
- `cumulative_score`
- `last_action_error`
- `tickets`
- `allowed_actions`

Each ticket contains sender, subject, body, customer tier, and mutable triage fields such as category, priority, queue, status, and latest reply.

### Action space

Defined as `SupportTriageAction`.

Supported `action_type` values:

- `classify`
- `prioritize`
- `assign`
- `reply`
- `set_status`
- `noop`

Depending on the action, the agent provides `ticket_id` plus one or more of:

- `category`
- `priority`
- `queue`
- `status`
- `message`

### Reward space

Defined as `SupportTriageReward` and implemented in [support_triage_env/env.py](/E:/projects/jarvis/support_triage_env/env.py).

Reward is shaped across the trajectory:

- positive reward for increasing the graded task score,
- a small reward for productive forward progress,
- penalties for invalid actions,
- penalties for repeating the same action signature too often.

This gives denser signal than a purely terminal pass/fail reward.

## Tasks and Graders

Tasks are defined in [support_triage_env/tasks.py](/E:/projects/jarvis/support_triage_env/tasks.py). Each task has a deterministic grader that scores work in `[0.0, 1.0]`.

### 1. `easy_refund_triage`

Difficulty: easy

Objective:
- Triage a single duplicate-charge refund request.

Expected behavior:
- classify as billing,
- set a sensible normal priority,
- assign to billing,
- send a reassuring reply,
- mark as pending.

### 2. `medium_inbox_triage`

Difficulty: medium

Objective:
- Triage a mixed inbox containing an outage, a refund case, and spam.

Expected behavior:
- route the outage to engineering with urgent priority,
- route the billing case to billing,
- mark spam correctly and send it to abuse.

### 3. `hard_enterprise_inbox`

Difficulty: hard

Objective:
- Triage an enterprise inbox containing a production outage, privacy request, billing dispute, and product request.

Expected behavior:
- distinguish between technical, security, billing, and product workflows,
- write replies with the right intent keywords,
- maintain correct queue and status decisions across multiple tickets.

### Grading breakdown

Each ticket is graded on:

- category
- priority
- queue
- status
- reply quality through required keywords

This is implemented deterministically in [support_triage_env/tasks.py](/E:/projects/jarvis/support_triage_env/tasks.py), which makes scores reproducible and easy to validate.

## Baseline Inference

The required inference script is:

- [inference.py](/E:/projects/jarvis/inference.py)

It:

- uses the OpenAI client for LLM calls,
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` / `OPENAI_API_KEY`,
- emits the required `[START]`, `[STEP]`, and `[END]` stdout format,
- falls back to a deterministic heuristic policy when no live model is available.

Current local baseline scores:

- `easy_refund_triage`: `1.000`
- `medium_inbox_triage`: `0.733`
- `hard_enterprise_inbox`: `0.613`

## Running Locally

Install dependencies:

```powershell
pip install -r requirements-openenv.txt
```

Run the server:

```powershell
python -m uvicorn support_triage_env.server:app --host 127.0.0.1 --port 7860
```

Run inference:

```powershell
python inference.py
```

Useful environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `OPENAI_API_KEY`
- `LOCAL_IMAGE_NAME`
- `IMAGE_NAME`
- `SUPPORT_TRIAGE_TASK`

## Docker

Build:

```powershell
docker build -t self-tooling-openenv .
```

Run:

```powershell
docker run --rm -p 7860:7860 self-tooling-openenv
```

## Hugging Face Spaces

This repo is prepared for a Docker-based Hugging Face Space.

Recommended Space settings:

- SDK: `Docker`
- Port: `7860`
- Tag: `openenv`

Expected health routes:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## OpenEnv Metadata

Environment metadata is defined in:

- [openenv.yaml](/E:/projects/jarvis/openenv.yaml)

Current API endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

## Project Structure

```text
support_triage_env/
  models.py
  tasks.py
  env.py
  server.py
  client.py
brain/
  decision_engine.py
  tool_synthesizer.py
inference.py
openenv.yaml
Dockerfile
requirements-openenv.txt
scripts/validate-submission.sh
```

## Validation

Pre-submission helper:

- [scripts/validate-submission.sh](/E:/projects/jarvis/scripts/validate-submission.sh)

Recommended checks:

```powershell
python inference.py
openenv validate
docker build -t self-tooling-openenv .
```

## Summary

`self-tooling-openenv` is both:

- a concrete OpenEnv benchmark for real customer support triage, and
- a broader research codebase showing how a self-improving agent can discover capability gaps, synthesize tools, validate them, and safely incorporate them into future behavior.

That combination is the core idea of the project: not only using tools, but learning when and how to create them.
