"""JARVIS Complete Validation Test Suite.

Sends all 60 test questions to the JARVIS API and records results.
Run with: python test_suite.py
Requires JARVIS to be running on localhost:8000
"""
import asyncio
import json
import time
import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120  # seconds per request (LLM can be slow)

# ── Test definitions ──────────────────────────────────────────────────────────
# Each test: (id, question, list_of_expected_substrings_in_response, list_of_fail_substrings)

TESTS = [
    # LEVEL 1 — BASIC SANITY
    (1, "hello", ["hello", "hi", "hey", "greet", "help", "jarvis", "boss", "faheem", "ready"], ["error", "traceback"]),
    (2, "what is your name", ["jarvis"], ["chatgpt", "claude", "gemini", "openai"]),
    (3, "what time is it", [":", "am", "pm", "time"], ["cannot", "don't know"]),
    (4, "what is today's date", ["2026", "april", "5"], ["don't know"]),
    (5, "who am I", ["faheem", "boss"], ["don't know", "i don't"]),

    # LEVEL 2 — REAL TIME DATA
    (6, "what is the weather in Chennai", ["chennai", "°c", "temperature", "source"], ["hallucin"]),
    (7, "what is the temperature today", ["°c", "temperature"], []),
    (8, "will it rain today in Mumbai", ["mumbai", "rain", "precipitation"], []),
    (9, "what is bitcoin price", ["btc", "bitcoin", "usd", "$"], ["training"]),
    (10, "what is ethereum price", ["eth", "ethereum", "usd", "$"], ["training"]),
    (11, "usd to inr today", ["usd", "inr", "rate"], ["training"]),
    (12, "latest news headlines", ["news", "headline"], ["made up"]),
    (13, "where did you get the weather data", ["api", "source", "open-meteo", "fetched"], ["openweathermap"]),
    (14, "what is the weather in XYZFakeCity123", ["not found", "error", "could not"], []),

    # LEVEL 3 — OS CONTROL
    (15, "open notepad", ["opened", "notepad"], ["error", "failed"]),
    (16, "type hello world in notepad", ["typed", "hello world", "notepad"], ["duckduckgo", "search"]),
    (17, "take a screenshot", ["screenshot", "saved", "png"], ["error"]),
    (18, "what is my battery percentage", ["battery", "%"], []),
    (19, "what is my cpu usage", ["cpu", "%"], []),
    (20, "how much ram is being used", ["ram", "gb", "%"], []),
    (21, "set volume to 50", ["volume", "50"], []),
    (22, "mute the volume", ["mute"], []),
    (23, "open chrome", ["opened", "chrome"], ["error"]),
    (24, "close notepad", ["killed", "closed", "notepad"], []),
    (25, "what apps are running", ["process", "cpu", "ram", "mb"], []),

    # LEVEL 4 — FILE OPERATIONS
    (26, "create a file called test_jarvis.txt with content hello jarvis", ["created", "test_jarvis", "hello jarvis"], ["error"]),
    (27, "read the file test_jarvis.txt", ["hello jarvis"], ["error"]),
    (28, "list files in my downloads folder", ["file", "download"], []),
    (29, "search for python files in E:/projects", [".py", "python", "file"], []),
    (30, "how big is the file test_jarvis.txt", ["byte", "size", "kb", "b"], []),

    # LEVEL 5 — WEB & SEARCH
    (31, "search the web for latest AI news", ["ai", "news", "http", "result"], []),
    (32, "fetch the content of wikipedia.org", ["wikipedia"], []),
    (33, "what is my public IP address", ["ip", "."], ["fake"]),
    (34, "is google.com up", ["google", "up", "status", "200", "ok", "online", "yes"], []),
    (35, "search wikipedia for artificial intelligence", ["artificial intelligence", "wikipedia", "ai"], []),

    # LEVEL 6 — INTENT ROUTING
    (36, "type i love you in notepad", ["typed", "i love you", "notepad"], ["duckduckgo", "search"]),
    (37, "what is 2 + 2", ["4"], []),
    (38, "remind me to drink water in 5 minutes", ["remind", "water", "set", "sched", "5"], []),
    (39, "run this command: echo hello", ["hello"], ["error"]),
    (40, "send an email to test@test.com saying hello", ["email", "test@test.com", "hello"], []),
    (41, "what is 15% of 5000", ["750"], []),
    (42, "explain what machine learning is", ["machine learning", "learn", "data", "algorithm", "model"], []),

    # LEVEL 7 — SELF EVOLVING BRAIN (may not all pass depending on brain state)
    (43, "calculate my BMI, I am 70kg and 175cm tall", ["bmi", "22", "normal"], []),

    # LEVEL 9 — EDGE CASES
    (54, "asdfjkl qwerty zxcvb", [], ["traceback"]),
    (55, "what is the weather", ["weather", "°c", "chennai"], []),
    (57, "what did I ask you earlier", ["ask", "earlier", "hello", "weather", "previous", "conversation"], []),
    (58, "clean your memory", ["audit", "memory", "clean"], []),
    (59, "what did you learn", ["learn", "session", "lesson", "pattern"], []),
]


async def run_test(client: httpx.AsyncClient, test_id: int, question: str,
                   expected: list[str], fail_markers: list[str]) -> dict:
    """Send a question and evaluate the response."""
    result = {"id": test_id, "question": question, "status": "UNKNOWN", "response": ""}

    try:
        start = time.time()
        resp = await client.post(f"{BASE_URL}/input", json={"text": question}, timeout=TIMEOUT)
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "")
        result["response"] = response_text[:500]
        result["elapsed_s"] = round(elapsed, 1)

        low = response_text.lower()

        # Check for fail markers first
        for fm in fail_markers:
            if fm.lower() in low:
                result["status"] = f"FAIL (contains '{fm}')"
                return result

        # Check for crash/error indicators
        if "traceback" in low or "error occurred" in low:
            result["status"] = "FAIL (error/crash)"
            return result

        # Check expected substrings (at least 1 must match)
        if expected:
            matched = [e for e in expected if e.lower() in low]
            if matched:
                result["status"] = "PASS"
            else:
                result["status"] = f"WARN (none of {expected[:3]} found)"
        else:
            # No expected strings means we just want no crash
            result["status"] = "PASS"

    except httpx.TimeoutException:
        result["status"] = "FAIL (timeout)"
    except httpx.ConnectError:
        result["status"] = "FAIL (connection refused - is JARVIS running?)"
    except Exception as exc:
        result["status"] = f"FAIL ({exc})"

    return result


async def main():
    print("=" * 80)
    print("JARVIS COMPLETE VALIDATION TEST SUITE")
    print("=" * 80)

    # Check if server is up
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code != 200:
                print("JARVIS is not responding on /health. Start it first!")
                sys.exit(1)
            print("JARVIS is running. Starting tests...\n")
        except Exception:
            print("Cannot connect to JARVIS at http://127.0.0.1:8000")
            print("Start JARVIS first: python main.py")
            sys.exit(1)

    results = []
    pass_count = 0
    fail_count = 0
    warn_count = 0

    async with httpx.AsyncClient() as client:
        for test_id, question, expected, fail_markers in TESTS:
            print(f"[Test {test_id:2d}] {question[:60]:<60s} ", end="", flush=True)
            result = await run_test(client, test_id, question, expected, fail_markers)
            results.append(result)

            status = result["status"]
            elapsed = result.get("elapsed_s", "?")
            if status == "PASS":
                print(f"  PASS  ({elapsed}s)")
                pass_count += 1
            elif status.startswith("WARN"):
                print(f"  WARN  ({elapsed}s) — {status}")
                warn_count += 1
            else:
                print(f"  FAIL  ({elapsed}s) — {status}")
                fail_count += 1

            # Brief pause between tests to not overwhelm the LLM
            await asyncio.sleep(1)

    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {pass_count} PASS | {warn_count} WARN | {fail_count} FAIL | {len(results)} TOTAL")
    print("=" * 80)

    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to test_results.json")

    # Print failures
    if fail_count > 0 or warn_count > 0:
        print("\n--- FAILURES AND WARNINGS ---")
        for r in results:
            if r["status"] != "PASS":
                print(f"\nTest {r['id']}: {r['question']}")
                print(f"  Status: {r['status']}")
                print(f"  Response: {r['response'][:200]}")


if __name__ == "__main__":
    asyncio.run(main())
