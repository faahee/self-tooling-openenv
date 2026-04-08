"""One-shot memory cleanup script.

Run once from the project root:
    python tools/clean_memory.py

What it does:
  1. Loads knowledge_graph.json + indexed_texts.json
  2. Filters out hallucinated / garbage / error entries
  3. Deduplicates near-identical entries (cosine ≥ 0.92)
  4. Rebuilds FAISS index with IndexFlatIP (cosine similarity)
  5. Saves everything back to disk
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
GRAPH_PATH   = ROOT / "data" / "knowledge_graph.json"
TEXTS_PATH   = ROOT / "data" / "indexed_texts.json"
FAISS_PATH   = ROOT / "data" / "faiss_index.bin"

# ── hallucination / garbage detection ─────────────────────────────────────────
# If ANY of these substrings appear in the JARVIS response → drop the entry.
HALLUCINATION_MARKERS = [
    "documentary", "gender studies", "nutritional plan", "meal plan",
    "hypothetinally", "hypothetically complex", "write a detailed narrative",
    "your task is to help me create", "the greatest hundred",
    "the wandering soul", "head chef", "sophimovie_idol",
    "the artificial intelligence researcher",
    "cognitive load theory",  # hallucinated UX essay
    "openShoe", "openshoe",
    "linux-based operating systems",  # wrong OS claim
    "2045", "2065", "2december",  # garbage dates
    "as of march 9, 2",  # garbled date
    "windows enterprises inc", "windows ai solutions",
    # error responses
    "couldn't determine the specific action",
]

# If ALL of these are true for an entry → likely hallucination:
#   - user input ≤ 4 words
#   - response length > 300 chars
SHORT_INPUT_LONG_RESPONSE_LIMIT = 300

# Response prefixes that indicate a failure
ERROR_PREFIXES = (
    "an error occurred", "error during", "i couldn't",
    "traceback", "exception:", "i understood this as an os command",
)

DEDUP_THRESHOLD = 0.92  # cosine similarity


# ── helpers ────────────────────────────────────────────────────────────────────

def is_hallucinated(user_input: str, response: str) -> bool:
    resp_lower = response.lower()
    # Explicit markers
    if any(m in resp_lower for m in HALLUCINATION_MARKERS):
        return True
    # Short input → very long rambling response
    if len(user_input.split()) <= 4 and len(response) > SHORT_INPUT_LONG_RESPONSE_LIMIT:
        return True
    # Error prefixes
    if any(resp_lower.startswith(p) for p in ERROR_PREFIXES):
        return True
    return False


def is_garbage_input(user_input: str) -> bool:
    """True for gibberish like 'hhihihnjkm' or repeated chars."""
    stripped = user_input.strip()
    if len(stripped) < 2:
        return True
    # high ratio of repeated characters relative to unique chars
    unique = len(set(stripped.lower()))
    if unique / max(len(stripped), 1) < 0.35 and len(stripped) > 4:
        return True
    return False


def parse_entry(node: dict) -> tuple[str, str]:
    """Extract (user_input, response) from a node text field."""
    text = node.get("text", "")
    if "\nJARVIS: " in text:
        parts = text.split("\nJARVIS: ", 1)
        user = parts[0].removeprefix("User: ").strip()
        resp = parts[1].strip()
    else:
        user = text.strip()
        resp = ""
    return user, resp


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    graph_data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    texts_data = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))

    all_nodes: list[dict] = graph_data.get("nodes", [])
    print(f"  Loaded {len(all_nodes)} graph nodes, {len(texts_data)} indexed texts")

    # ── Step 1: filter bad entries ─────────────────────────────────────────────
    kept_nodes: list[dict] = []
    dropped = 0
    for node in all_nodes:
        user, resp = parse_entry(node)
        if is_garbage_input(user):
            print(f"  [GARBAGE INPUT] {user[:40]!r}")
            dropped += 1
            continue
        if is_hallucinated(user, resp):
            print(f"  [HALLUCINATED]  {user[:40]!r} -> {resp[:60]!r}...")
            dropped += 1
            continue
        kept_nodes.append(node)

    print(f"\n  Dropped {dropped} bad entries, kept {len(kept_nodes)}")

    # ── Step 2: re-embed and deduplicate ──────────────────────────────────────
    print("\nLoading embedder (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    texts_to_embed = [n.get("text", "") for n in kept_nodes]
    print(f"Embedding {len(texts_to_embed)} entries...")
    embeddings = model.encode(
        texts_to_embed, normalize_embeddings=True,
        show_progress_bar=True, batch_size=32,
    )

    # Greedy deduplication: keep first, skip subsequent that are too close
    index_ip = faiss.IndexFlatIP(384)
    final_nodes: list[dict] = []
    final_texts: list[dict] = []
    deduped = 0

    for i, (node, emb) in enumerate(zip(kept_nodes, embeddings)):
        emb_arr = np.array([emb], dtype=np.float32)
        if index_ip.ntotal > 0:
            sims, _ = index_ip.search(emb_arr, 1)
            if sims[0][0] >= DEDUP_THRESHOLD:
                user, _ = parse_entry(node)
                print(f"  [DEDUP sim={sims[0][0]:.3f}] {user[:50]!r}")
                deduped += 1
                continue
        index_ip.add(emb_arr)
        final_nodes.append(node)
        # Rebuild indexed_texts entry
        orig = next(
            (t for t in texts_data if t.get("node_id") == node.get("id")),
            {"text": node.get("text", ""), "category": node.get("category", "interaction"),
             "metadata": node.get("metadata", {}), "node_id": node.get("id", "")},
        )
        final_texts.append(orig)

    print(f"\n  Removed {deduped} duplicates")
    print(f"  Final clean entries: {len(final_nodes)}")

    # ── Step 3: save ──────────────────────────────────────────────────────────
    print("\nSaving...")
    graph_data["nodes"] = final_nodes
    graph_data["links"] = []  # edges are rebuilt by entity extraction going forward
    GRAPH_PATH.write_text(json.dumps(graph_data, indent=2, default=str), encoding="utf-8")

    TEXTS_PATH.write_text(json.dumps(final_texts, indent=2, default=str), encoding="utf-8")

    faiss.write_index(index_ip, str(FAISS_PATH))

    print(f"  knowledge_graph.json — {len(final_nodes)} nodes")
    print(f"  indexed_texts.json   — {len(final_texts)} entries")
    print(f"  faiss_index.bin      — {index_ip.ntotal} vectors (IndexFlatIP / cosine)")
    print("\nDone! Memory is clean.")


if __name__ == "__main__":
    main()
