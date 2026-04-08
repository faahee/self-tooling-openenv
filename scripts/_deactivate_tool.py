"""Update get_largest_files tool description and embedding, reset stats."""
import sqlite3
import json
from sentence_transformers import SentenceTransformer

new_desc = "find the biggest or largest files on a drive, scan disk for large files"
name = "get_largest_files"
name_words = name.replace("_", " ")
embed_text = f"{name_words}: {new_desc}"

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(embed_text)
embedding_json = json.dumps(embedding.tolist())

conn = sqlite3.connect("data/brain.db")
conn.execute(
    "UPDATE tools SET description=?, description_embedding=?, success_rate=1.0, total_runs=1 WHERE name=?",
    (new_desc, embedding_json, name),
)
conn.commit()
print(f"Updated {name}: desc='{new_desc}'")
conn.close()
