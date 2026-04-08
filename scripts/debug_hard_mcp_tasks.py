"""Run harder Windows-MCP-backed tasks through the orchestrator."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug_query_batch import build_orchestrator


TEST_CASES = [
    {
        "query": 'run powershell command Write-Output "jarvis-hard-mcp-ok"',
        "must_contain": ["jarvis-hard-mcp-ok"],
    },
    {
        "query": "list running processes",
        "must_contain": ["process", "pid"],
    },
    {
        "query": "copy jarvis hard mcp test to clipboard",
        "must_contain": ["clipboard"],
    },
    {
        "query": "what is in my clipboard",
        "must_contain": ["jarvis hard mcp test"],
    },
    {
        "query": "send a windows notification saying jarvis notification test",
        "must_contain": ["notification"],
    },
    {
        "query": "inspect the current screen",
        "must_contain": ["cursor position"],
    },
    {
        "query": "open calculator",
        "must_contain": ["opened", "calculator"],
    },
    {
        "query": "switch to calculator window",
        "must_contain": ["calculator"],
    },
]


async def main() -> None:
    orchestrator, classifier, mcp_client = await build_orchestrator()
    try:
        print("=== Hard MCP Task Debug ===")
        print(f"windows_mcp_connected={any(s.connected for s in mcp_client.servers.values())}")
        print(f"windows_mcp_tools={len(mcp_client.get_all_tools())}")
        print()

        passes = 0
        failures: list[tuple[str, str]] = []
        for idx, case in enumerate(TEST_CASES, 1):
            query = case["query"]
            intent, confidence = classifier.classify(query)
            response = await orchestrator.handle_input(query)
            low = response.lower()
            issues = [marker for marker in case["must_contain"] if marker.lower() not in low]
            ok = not issues
            if ok:
                passes += 1
            else:
                failures.append((query, ", ".join(issues)))

            print(f"[{idx:02d}] {query}")
            print(f"  classify={intent} ({confidence:.2f})")
            print(f"  response={response[:300]}")
            print(f"  verdict={'PASS' if ok else 'FAIL'}")
            if issues:
                print(f"  issues=missing {', '.join(issues)}")
            print()

        print(f"SUMMARY: {passes}/{len(TEST_CASES)} passed")
        if failures:
            print("FAILURES:")
            for query, issues in failures:
                print(f"- {query}: {issues}")
    finally:
        if mcp_client.enabled:
            await mcp_client.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
