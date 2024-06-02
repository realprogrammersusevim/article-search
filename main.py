import datetime
import pickle
import sqlite3

import sentence_transformers
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from utils import create_index, Article, summarize_article

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def hello_world():
    return render_template("index.html")


@socketio.on("search")
def handle_search(search):
    print(search)

    # Set up the embedding model
    model = sentence_transformers.SentenceTransformer(
        "Snowflake/snowflake-arctic-embed-m", device="mps"
    )
    emit("status", "Loaded model")
    # Load the data
    with open("dump.pickle", "rb") as f:
        articles = pickle.load(f)
        print(articles[0])

    emit("status", "Loaded data")

    # Embed the search query
    search_embed = model.encode(search)
    emit("status", "Embedded search query")

    # Calculate the similarity between the search query and each article
    for i, art in enumerate(articles):
        embed = art.embedding
        similarity = sentence_transformers.util.cos_sim(embed, search_embed)
        art.similarity = similarity

    emit("status", "Calculated similarities")

    # Connect to the database
    conn = sqlite3.connect("news.db")
    cur = conn.cursor()

    # Sort the articles by similarity
    articles = sorted(articles, key=lambda x: x.similarity, reverse=True)

    # Get the top 10 articles
    articles = articles[:10]

    index = create_index()

    searcher = index.searcher()
    query = index.parse_query(search, ["title", "body"])

    (top_docs, top_addresses) = searcher.search(query, 10).hits[:10]
    best_docs = [searcher.doc(address) for address in top_addresses]

    # Load the cross encoder
    model = sentence_transformers.SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1"
    )

    search_embed = model.encode(
        f"Represent this sentence for searching relevant news articles: {search}"
    )
    synthesized = articles
    for art in best_docs:
        synthesized.append(
                Article(art["title"], art["body"], art["id"])
        )
    embeddings = model.encode([art.body for art in synthesized])
    for i, article in enumerate(synthesized):
        article.embedding = embeddings[i]

    # Calculate the similarity between the search query and each article
    for i, art in enumerate(synthesized):
        embed = art.embedding
        similarity = sentence_transformers.util.cos_sim(embed, search_embed)
        art.similarity = similarity

    # Sort the articles by similarity
    synthesized = sorted(synthesized, key=lambda x: x.similarity, reverse=True)

    # Get the top 10 articles
    synthesized = synthesized[:10]

    for article in synthesized:
        # The article we're working with
        print(article.title)
        emit("status", f"Processing article {article.title}")

        cur.execute("SELECT * FROM articles_meta WHERE uid = ?", (article.id,))
        meta = cur.fetchone()
        print(meta)
        formatted = int(str(meta[2])[:-3])  # Chop off the last three digits
        date = datetime.datetime.fromtimestamp(formatted, datetime.UTC)

        # Add the article's metadata
        article.date = date.strftime("%a, %B %d %Y")
        article.url = meta[3]
        article.publication = meta[5]

        # Change article to be JSON serializable
        article.embedding = None
        article.similarity = None

        # Send the article to the client
        print("Sending article")
        emit("status", "Sending article")
        emit("new_article", article.serializable())

        print("Got metadata, starting generation...")
        emit("status", "Got metadata, starting generation...")

        summarize_article(article)


if __name__ == "__main__":
    socketio.run(app)
