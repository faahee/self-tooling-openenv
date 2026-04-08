"""Test: capture full Step 2 summary from workflow."""
import httpx, asyncio

async def main():
    async with httpx.AsyncClient(timeout=300) as c:
        resp = await c.post(
            "http://127.0.0.1:8000/input",
            json={"text": "get latest war news and summarize into a report"},
        )
        data = resp.json()
        text = data.get("response", "")
        
        # Extract Step 2 content
        lines = text.split("\n")
        in_step2 = False
        step2_content = []
        for line in lines:
            if "Step 2" in line:
                in_step2 = True
                continue
            if in_step2 and "Step 3" in line:
                break
            if in_step2:
                step2_content.append(line)
        
        print("=" * 80)
        print("STEP 2 — FULL SUMMARY:")
        print("=" * 80)
        summary = "\n".join(step2_content).strip()
        if summary:
            print(summary)
        else:
            print("(Could not extract Step 2; printing full response)")
            print(text)
        print("=" * 80)
        
        # Freshness analysis
        print("\n--- FRESHNESS ANALYSIS ---")
        stale = ["October 2023", "February 2022", "500,000 lives", "40,000+ killed", 
                 "150,000+ deaths", "April 2023"]
        current = ["2025", "2026", "Iran-US", "Trump", "Iran war", "Hormuz"]
        
        found_stale = [m for m in stale if m.lower() in text.lower()]
        found_current = [m for m in current if m.lower() in text.lower()]
        
        for m in found_stale:
            print(f"  ⚠️  OLD data: '{m}'")
        for m in found_current:
            print(f"  ✅ CURRENT: '{m}'")
        
        if found_current and not found_stale:
            print("\n  ✅ VERDICT: Report uses CURRENT news only!")
        elif found_current and found_stale:
            print(f"\n  ⚠️  VERDICT: Mix of current + old data ({len(found_stale)} stale markers)")
        elif found_stale:
            print("\n  ❌ VERDICT: Report uses OLD training data")
        else:
            print("\n  ℹ️  VERDICT: No date markers to judge")

asyncio.run(main())
