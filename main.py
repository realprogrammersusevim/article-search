import sentence_transformers
import sqlite3
import numpy as np

# Load the pre-trained model
model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2", device="mps")

# Connect to the SQLite database
conn = sqlite3.connect("news.db")
cur = conn.cursor()

cur.execute("SELECT * FROM articles_fts;")
articles = []
skipped = 0
for i, article in enumerate(cur):
    content = article[1]

    if content == "":
        skipped += 1
    else:
        articles.append(content)

print(f"Skipped {skipped} articles")
print(f"Generating embeddings for {len(articles)} articles")
embeddings = model.encode(articles, show_progress_bar=True)
search = "What should the U. S. do about Israel and Iran?"
search_embed = model.encode(search)
similarities = []
for i, embed in enumerate(embeddings):
    similarity = np.dot(embed, search_embed) / (np.linalg.norm(embed) * np.linalg.norm(search_embed))
    similarities.append({"embed": embed, "similarity": similarity, "article": articles[i]})

sorted = sorted(similarities, key=lambda x: x["similarity"])
for i in sorted[-10:]:
    print(i["article"])
