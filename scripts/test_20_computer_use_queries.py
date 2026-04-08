"""Run a mixed 20-query computer-use validation batch."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.workflow_agent import WorkflowAgent
from scripts.debug_query_batch import build_orchestrator


TEST_CASES = [
    {"query": "open browser", "intent": "OS_COMMAND", "must": ["opened", "chrome"]},
    {"query": "open calculator", "intent": "OS_COMMAND", "must": ["opened", "calculator"]},
    {"query": "minimize all screens", "intent": "OS_COMMAND", "must": ["desktop", "minimized"]},
    {"query": "show me the desktop", "intent": "OS_COMMAND", "must": ["desktop", "minimized"]},
    {"query": "turn up volume", "intent": "OS_COMMAND", "must": ["volume"]},
    {"query": "mute the volume", "intent": "OS_COMMAND", "must": ["muted"]},
    {"query": "unmute the volume", "intent": "OS_COMMAND", "must": ["unmuted"]},
    {"query": "pause", "intent": "OS_COMMAND", "must": ["pause", "play"]},
    {"query": "next song", "intent": "OS_COMMAND", "must": ["next song", "skipped"]},
    {"query": "take screenshot", "intent": "OS_COMMAND", "must": ["screenshot"]},
    {"query": 'run powershell command Write-Output "jarvis-20-ok"', "intent": "OS_COMMAND", "must": ["jarvis-20-ok"]},
    {"query": "list running processes", "intent": "OS_COMMAND", "must": ["process", "pid"]},
    {"query": "copy jarvis 20 query test to clipboard", "intent": "OS_COMMAND", "must": ["clipboard"]},
    {"query": "what is in my clipboard", "intent": "OS_COMMAND", "must": ["jarvis 20 query test"]},
    {"query": "send a windows notification saying jarvis twenty test", "intent": "OS_COMMAND", "must": ["notification"]},
    {"query": "inspect the current screen", "intent": "OS_COMMAND", "must": ["cursor position"]},
    {"query": "switch to calculator window", "intent": "OS_COMMAND", "must": ["calculator"]},
    {"query": "where is java file", "intent": "FILE_OP", "must": ["file"]},
    {"query": "where is my document", "intent": "FILE_OP", "must": ["file"]},
    {"query": "create a folder name kool in desktop", "intent": "FILE_OP", "must": ["created folder", "desktop", "kool"]},
]


def evaluate(case: dict, classified_intent: str, response: str) -> list[str]:
    issues: list[str] = []
    if classified_intent != case["intent"]:
        issues.append(f"classified as {classified_intent}, expected {case['intent']}")
    low = response.lower()
    for marker in case["must"]:
        if marker.lower() not in low:
            issues.append(f"missing expected marker: {marker}")
    if "[terminal_fallback]" in low:
        issues.append("fell back to terminal agent")
    if "i need live data for that" in low:
        issues.append("misrouted to live-data response")
    if "workflow complete" in low and case["intent"] != "WORKFLOW":
        issues.append("misrouted into workflow")
    return issues


async def main() -> None:
    orchestrator, classifier, mcp_client = await build_orchestrator()
    try:
        print("=== 20 Query Computer-Use Batch ===")
        print(f"windows_mcp_connected={any(s.connected for s in mcp_client.servers.values())}")
        print(f"windows_mcp_tools={len(mcp_client.get_all_tools())}")
        print()

        passes = 0
        failures: list[dict] = []
        for idx, case in enumerate(TEST_CASES, 1):
            query = case["query"]
            intent, confidence = classifier.classify(query)
            workflow = WorkflowAgent.is_workflow(query)
            response = await orchestrator.handle_input(query)
            issues = evaluate(case, intent, response)
            ok = not issues
            if ok:
                passes += 1
            else:
                failures.append({"query": query, "issues": issues, "response": response})

            print(f"[{idx:02d}] {query}")
            print(f"  classify={intent} ({confidence:.2f}) workflow={workflow}")
            print(f"  response={response[:260]}")
            print(f"  verdict={'PASS' if ok else 'FAIL'}")
            if issues:
                print(f"  issues={' | '.join(issues)}")
            print()

        print(f"SUMMARY: {passes}/{len(TEST_CASES)} passed")
        if failures:
            print("FAILURES:")
            for failure in failures:
                print(f"- {failure['query']}: {' | '.join(failure['issues'])}")
    finally:
        if mcp_client.enabled:
            await mcp_client.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
