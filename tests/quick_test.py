import httpx, asyncio, sys

async def test(q):
    async with httpx.AsyncClient() as c:
        resp = await c.post("http://127.0.0.1:8000/input", json={"text": q}, timeout=60)
        r = resp.json()
        print(f"Q: {q}")
        print(f"A: {r.get('response','')[:300]}")
        print()

async def main():
    questions = [
        "fetch the content of wikipedia.org",
        "is google.com up",
        "search wikipedia for artificial intelligence",
    ]
    for q in questions:
        await test(q)

asyncio.run(main())
