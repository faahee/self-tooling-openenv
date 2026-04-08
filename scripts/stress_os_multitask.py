"""Run a concurrent 10-command Windows-style stress test against the orchestrator."""
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
    {
        "query": "open browser",
        "intent": "OS_COMMAND",
        "must_contain": ["opened", "chrome"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "open calculator",
        "intent": "OS_COMMAND",
        "must_contain": ["opened", "calculator"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "minimize all screens",
        "intent": "OS_COMMAND",
        "must_contain": ["desktop", "minimized"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "show me the desktop",
        "intent": "OS_COMMAND",
        "must_contain": ["desktop", "minimized"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "turn up volume",
        "intent": "OS_COMMAND",
        "must_contain": ["volume"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "mute the volume",
        "intent": "OS_COMMAND",
        "must_contain": ["mute"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "unmute the volume",
        "intent": "OS_COMMAND",
        "must_contain": ["unmuted"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "pause",
        "intent": "OS_COMMAND",
        "must_contain": ["pause", "play"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "next song",
        "intent": "OS_COMMAND",
        "must_contain": ["next song", "skipped"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
    {
        "query": "take screenshot",
        "intent": "OS_COMMAND",
        "must_contain": ["screenshot"],
        "must_not": ["[terminal_fallback]", "I need live data for that"],
    },
]


def evaluate(case: dict, classified_intent: str, response: str) -> list[str]:
    issues: list[str] = []
    if classified_intent != case["intent"]:
        issues.append(f"classified as {classified_intent}, expected {case['intent']}")
    low = response.lower()
    for marker in case.get("must_not", []):
        if marker.lower() in low:
            issues.append(f"response contains forbidden marker: {marker}")
    for marker in case.get("must_contain", []):
        if marker.lower() not in low:
            issues.append(f"response missing expected marker: {marker}")
    return issues


async def run_one(orchestrator, classifier, case: dict) -> dict:
    query = case["query"]
    classified_intent, confidence = classifier.classify(query)
    workflow = WorkflowAgent.is_workflow(query)
    response = await orchestrator.handle_input(query)
    return {
        "query": query,
        "classified_intent": classified_intent,
        "confidence": confidence,
        "workflow": workflow,
        "response": response,
        "issues": evaluate(case, classified_intent, response),
    }


async def main() -> None:
    orchestrator, classifier, mcp_client = await build_orchestrator()
    try:
        print("=== Concurrent OS Stress Test ===")
        print(f"windows_mcp_connected={any(s.connected for s in mcp_client.servers.values())}")
        print(f"windows_mcp_tools={len(mcp_client.get_all_tools())}")
        print(f"concurrency={len(TEST_CASES)}")
        print()

        results = await asyncio.gather(*[run_one(orchestrator, classifier, case) for case in TEST_CASES])

        passes = 0
        for idx, result in enumerate(results, 1):
            ok = not result["issues"]
            if ok:
                passes += 1
            print(f"[{idx:02d}] {result['query']}")
            print(
                f"  classify={result['classified_intent']} ({result['confidence']:.2f}) "
                f"workflow={result['workflow']}"
            )
            print(f"  response={result['response'][:240]}")
            print(f"  verdict={'PASS' if ok else 'FAIL'}")
            if result["issues"]:
                print(f"  issues={' | '.join(result['issues'])}")
            print()

        print(f"SUMMARY: {passes}/{len(results)} passed")
        failures = [r for r in results if r["issues"]]
        if failures:
            print("FAILURES:")
            for failure in failures:
                print(f"- {failure['query']}: {' | '.join(failure['issues'])}")
    finally:
        if mcp_client.enabled:
            await mcp_client.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
