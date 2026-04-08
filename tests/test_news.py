"""Test: war news workflow — check if output uses current news vs training data."""
import httpx, asyncio, json

async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        resp = await c.post(
            "http://127.0.0.1:8000/input",
            json={"text": "i want latest news of war and put them in the pdf"},
        )
        data = resp.json()
        text = data.get("response", "")

        print("=" * 80)
        print("FULL RESPONSE:")
        print("=" * 80)
        print(text)
        print("=" * 80)

        # Check for staleness indicators
        stale_markers = [
            "October 2023",       # training data date
            "February 2022",      # training data event start
            "500,000 lives",      # training data figure
            "40,000+ killed",     # training data figure
            "150,000+ deaths",    # training data figure
        ]
        current_markers = [
            "2025", "2026",       # current year references
        ]

        print("\n--- STALENESS CHECK ---")
        for m in stale_markers:
            if m.lower() in text.lower():
                print(f"  ⚠️  STALE marker found: '{m}'")
        for m in current_markers:
            if m.lower() in text.lower():
                print(f"  ✅ CURRENT marker found: '{m}'")

        # Check if it's just regurgitating old facts
        if any(m.lower() in text.lower() for m in stale_markers) and not any(m.lower() in text.lower() for m in current_markers):
            print("\n  ❌ VERDICT: Output appears to use OLD training data, not current news")
        elif any(m.lower() in text.lower() for m in current_markers):
            print("\n  ✅ VERDICT: Output contains current-year references")
        else:
            print("\n  ℹ️  VERDICT: No clear date markers — review content manually")

asyncio.run(main())
