"""Memory System — Knowledge graph + FAISS vector store for persistent memory."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MemorySystem:
    """Persistent long-term memory using a NetworkX knowledge graph and FAISS vector index.

    Every interaction makes JARVIS smarter about the user.
    """

    def __init__(self, config: dict, embedder=None) -> None:
        """Initialize the memory system.

        Args:
            config: Full JARVIS configuration dictionary.
            embedder: Optional shared SentenceTransformer instance.
        """
        self.graph_path = Path(config["memory"]["graph_path"])
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        self._texts_path = self.graph_path.parent / "indexed_texts.json"
        self._faiss_path = self.graph_path.parent / "faiss_index.bin"

        if embedder is not None:
            self.embedder = embedder
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.graph = nx.DiGraph()
        # IndexFlatIP = inner product (= cosine similarity when embeddings are L2-normalized)
        self.faiss_index = faiss.IndexFlatIP(384)  # 384 = MiniLM embedding dim
        self.indexed_texts: list[dict] = []  # parallel list to faiss_index
        self._load_from_disk()
        # Ensure the root user node always exists
        if not self.graph.has_node("Faheem"):
            self.graph.add_node("Faheem", type="user")

    # ── Public API ────────────────────────────────────────────────────────────

    # Minimum cosine distance to nearest existing memory before we store a new one.
    # Prevents the index from filling up with near-identical interactions.
    _DEDUP_THRESHOLD = 0.95

    def add_memory(
        self,
        text: str,
        category: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a memory to both the knowledge graph and vector index.

        Near-duplicate entries (cosine > 0.95) are silently skipped so the
        index stays lean and query results stay diverse.

        Args:
            text: The text content of the memory.
            category: One of "interaction", "fact", "goal", "project", "preference".
            metadata: Optional extra metadata dict.
        """
        embedding = self.embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        emb_arr = np.array(embedding, dtype=np.float32)

        # Deduplication: skip if very similar entry already exists
        if self.faiss_index.ntotal > 0:
            distances, _ = self.faiss_index.search(emb_arr, 1)
            if distances[0][0] >= self._DEDUP_THRESHOLD:
                logger.debug("Memory deduped (sim=%.3f): %s", distances[0][0], text[:60])
                return

        node_id = f"mem_{len(self.graph.nodes)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.graph.add_node(
            node_id,
            text=text,
            category=category,
            metadata=metadata or {},
            created=datetime.now().isoformat(),
        )

        self.faiss_index.add(emb_arr)
        self.indexed_texts.append(
            {"text": text, "category": category, "metadata": metadata or {}, "node_id": node_id}
        )
        self._save_to_disk()

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant memories for a query.

        Args:
            query_text: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, category, metadata, similarity_score.
        """
        if self.faiss_index.ntotal == 0:
            return []

        embedding = self.embedder.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
        k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(
            np.array(embedding, dtype=np.float32), k
        )

        results: list[dict] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.indexed_texts):
                continue
            entry = self.indexed_texts[idx]
            results.append(
                {
                    "text": entry["text"],
                    "category": entry["category"],
                    "metadata": entry["metadata"],
                    "similarity_score": float(score),  # already in [0,1] (cosine)
                }
            )
        return results

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Add a structured fact to the knowledge graph.

        Args:
            subject: The subject entity.
            predicate: The relationship.
            obj: The object entity.
        """
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(obj, type="entity")
        self.graph.add_edge(subject, obj, relation=predicate)

        fact_text = f"{subject} {predicate} {obj}"
        self.add_memory(fact_text, category="fact", metadata={"subject": subject, "predicate": predicate, "object": obj})

    def get_related(self, entity: str, depth: int = 2) -> list[dict]:
        """BFS from entity node up to *depth* hops.

        Args:
            entity: Starting entity node.
            depth: Maximum hops.

        Returns:
            List of related fact dicts.
        """
        if entity not in self.graph:
            return []

        related: list[dict] = []
        visited = set()
        queue = [(entity, 0)]
        while queue:
            node, d = queue.pop(0)
            if node in visited or d > depth:
                continue
            visited.add(node)
            for _, neighbor, data in self.graph.edges(node, data=True):
                related.append(
                    {"from": node, "relation": data.get("relation", ""), "to": neighbor}
                )
                if d + 1 <= depth:
                    queue.append((neighbor, d + 1))
        return related

    # Prefixes that indicate JARVIS failed — don't memorize failures
    _ERROR_PREFIXES = (
        "an error occurred", "error during", "i couldn't", "sorry, i",
        "sorry i", "couldn't determine", "failed to", "traceback",
        "exception:", "nothing to repeat",
    )

    def update(self, user_input: str, response: str) -> None:
        """Store a meaningful interaction as memory, skipping failures.

        Also extracts explicit user preferences (e.g. "remember that I use...")
        as standalone fact nodes for faster retrieval.

        Args:
            user_input: What the user said.
            response: What JARVIS said.
        """
        response_lower = response.strip().lower()
        # Skip error/fallback responses — they add noise, not signal
        if any(response_lower.startswith(p) for p in self._ERROR_PREFIXES):
            return
        # Skip very short non-informative responses
        if len(response.strip()) < 10:
            return

        combined = f"User: {user_input}\nJARVIS: {response[:300]}"
        self.add_memory(combined, category="interaction")

        # Extract explicit user preferences: "remember that …", "I use …", "I prefer …"
        low = user_input.lower()
        pref_triggers = ("remember that", "i use ", "i prefer ", "i like ", "my name is ", "i am ")
        for trigger in pref_triggers:
            if trigger in low:
                self.add_memory(user_input, category="preference")
                break

        # Build graph edges from this interaction
        self._extract_entities(user_input, response)

    # ── Entity extraction ─────────────────────────────────────────────────────

    # Known apps we track as entities
    _KNOWN_APPS = {
        "whatsapp", "telegram", "notepad", "chrome", "google chrome",
        "brave", "spotify", "discord", "steam", "epic games", "epic games launcher",
        "obs", "vlc", "zoom", "vs code", "vscode", "visual studio code",
        "file explorer", "calculator", "settings", "notepad++",
        "edge", "firefox", "slack", "teams", "word", "excel", "powerpoint",
    }

    # Verbs that indicate what JARVIS did successfully
    _ACTION_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("opened",   re.compile(r"^opened\s+(.+?)\.", re.I)),
        ("sent",     re.compile(r"^sent\s+['\"]?(.+?)['\"]?\s+to\s+(\w+)\s+on\s+(\w+)", re.I)),
        ("searched", re.compile(r"^searched\s+for\s+['\"]?(.+?)['\"]?\s+in\s+(\w+)", re.I)),
        ("typed",    re.compile(r"^opened\s+\S+\s+and\s+typed:\s+(.+)", re.I)),
        ("installed",re.compile(r"^installed\s+(.+?)\.", re.I)),
        ("ran",      re.compile(r"^ran\s+(.+?)\.", re.I)),
    ]

    # Contact extraction: "send X to NAME on APP", "search for NAME in APP"
    _CONTACT_SEND_RE = re.compile(
        r'(?:send|message|tell|write)\s+.+?\s+to\s+([a-zA-Z][a-zA-Z0-9_\- ]+?)'
        r'\s+(?:on|via|in|using)\s+(\w+)',
        re.I,
    )
    _CONTACT_SEARCH_RE = re.compile(
        r'(?:search|find|look\s*up)\s+(?:for\s+)?([a-zA-Z][a-zA-Z0-9_\- ]+?)'
        r'\s+(?:in|on)\s+(\w+)',
        re.I,
    )

    def _extract_entities(self, user_input: str, response: str) -> None:
        """Extract entities and relationships from an interaction and add graph edges.

        Creates nodes for: user ("Faheem"), apps, contacts.
        Creates edges like: Faheem -[used]-> whatsapp, Faheem -[contacted]-> rifana.
        """
        user_node = "Faheem"
        low_input = user_input.lower()
        low_resp = response.lower().strip()
        dirty = False

        # ── Detect apps mentioned by user ─────────────────────────────────────
        for app in self._KNOWN_APPS:
            if app in low_input:
                app_node = app.title()
                if not self.graph.has_node(app_node):
                    self.graph.add_node(app_node, type="app")
                if not self.graph.has_edge(user_node, app_node, key="used"):
                    self.graph.add_edge(user_node, app_node, relation="used")
                    logger.debug("Graph edge: %s -[used]-> %s", user_node, app_node)
                    dirty = True

        # ── Detect contacts (messaging) ───────────────────────────────────────
        for pattern, label in [
            (self._CONTACT_SEND_RE, "contacted"),
            (self._CONTACT_SEARCH_RE, "searched_for"),
        ]:
            m = pattern.search(user_input)
            if m:
                contact = m.group(1).strip().title()
                app_raw = m.group(2).strip().lower()
                app_node = app_raw.title()
                contact_node = contact

                if not self.graph.has_node(contact_node):
                    self.graph.add_node(contact_node, type="contact")
                if not self.graph.has_node(app_node):
                    self.graph.add_node(app_node, type="app")

                # Faheem -[contacted/searched_for]-> contact (via app)
                if not self.graph.has_edge(user_node, contact_node, key=label):
                    self.graph.add_edge(user_node, contact_node, relation=label, via=app_node)
                    logger.debug("Graph edge: %s -[%s]-> %s via %s", user_node, label, contact_node, app_node)
                    dirty = True
                # contact -[on]-> app
                if not self.graph.has_edge(contact_node, app_node, key="on"):
                    self.graph.add_edge(contact_node, app_node, relation="on")
                    dirty = True

        # ── Detect successful JARVIS actions from response ────────────────────
        for action, pattern in self._ACTION_PATTERNS:
            m = pattern.match(low_resp)
            if m:
                target = m.group(1).strip().title()
                action_node = f"{action}:{target}"
                if not self.graph.has_node(action_node):
                    self.graph.add_node(action_node, type="action", verb=action, target=target)
                if not self.graph.has_edge(user_node, action_node, key=action):
                    self.graph.add_edge(user_node, action_node, relation=action)
                    logger.debug("Graph edge: %s -[%s]-> %s", user_node, action, target)
                    dirty = True
                break

        if dirty:
            self._save_to_disk()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        """Persist graph, FAISS index, and indexed texts to disk."""
        try:
            data = json_graph.node_link_data(self.graph, edges="links")
            self.graph_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

            if self.faiss_index.ntotal > 0:
                faiss.write_index(self.faiss_index, str(self._faiss_path))

            self._texts_path.write_text(
                json.dumps(self.indexed_texts, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Failed to save memory to disk: %s", exc)

    def _load_from_disk(self) -> None:
        """Load existing data from disk or create empty structures."""
        # Graph
        if self.graph_path.exists():
            try:
                raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
                self.graph = json_graph.node_link_graph(raw, directed=True, edges="links")
            except Exception as exc:
                logger.warning("Could not load knowledge graph, starting fresh: %s", exc)
                self.graph = nx.DiGraph()

        # FAISS index
        if self._faiss_path.exists():
            try:
                loaded = faiss.read_index(str(self._faiss_path))
                # Migrate L2 index to IP index if needed (old format)
                if isinstance(loaded, faiss.IndexFlatL2):
                    logger.info("Migrating FAISS index from L2 to IP (cosine)...")
                    self.faiss_index = faiss.IndexFlatIP(384)
                    if loaded.ntotal > 0:
                        vecs = faiss.rev_swig_ptr(loaded.get_xb(), loaded.ntotal * 384)
                        vecs = np.array(vecs, dtype=np.float32).reshape(loaded.ntotal, 384)
                        # Re-normalize just in case, then add
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        self.faiss_index.add(vecs / norms)
                else:
                    self.faiss_index = loaded
            except Exception as exc:
                logger.warning("Could not load FAISS index, starting fresh: %s", exc)
                self.faiss_index = faiss.IndexFlatIP(384)

        # Indexed texts
        if self._texts_path.exists():
            try:
                self.indexed_texts = json.loads(self._texts_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load indexed texts, starting fresh: %s", exc)
                self.indexed_texts = []
