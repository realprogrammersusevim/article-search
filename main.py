import logging
import torch
import datetime
import pickle
import sqlite3
from typing import List

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
    # Set up the embedding model
    torch.device("mps")
    model = sentence_transformers.SentenceTransformer(
        "Snowflake/snowflake-arctic-embed-m"
    )
    emit("status", "Loaded model")
    # Load the data
    with open("dump.pickle", "rb") as f:
        articles = pickle.load(f)

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

    top_docs = searcher.search(query, 10).hits[:10]
    best_docs = [searcher.doc(doc[1]) for doc in top_docs]

    # Load the cross encoder
    model = sentence_transformers.SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1"
    )

    search_embed = model.encode(
        f"Represent this sentence for searching relevant news articles: {search}"
    )
    synthesized = articles
    for art in best_docs:
        synthesized.append(Article(art["title"][0], art["body"][0], art["id"][0]))
    embeddings = model.encode([art.body for art in synthesized])
    for i, article in enumerate(synthesized):
        article.embedding = embeddings[i]

    # Deduplicate the list
    synthesized = list(set(synthesized))

    # Calculate the similarity between the search query and each article
    for art in synthesized:
        embed = art.embedding
        similarity = sentence_transformers.util.cos_sim(embed, search_embed)
        art.similarity = similarity

    # Sort the articles by similarity
    synthesized = sorted(synthesized, key=lambda x: x.similarity, reverse=True)

    # Get the top 10 articles
    synthesized = synthesized[:10]

    for article in synthesized:
        # The article we're working with
        emit("status", f"Processing article {article.title}")

        if isinstance(article.id, List):
            article.id = article.id[0]  # Not sure why this is a list
        cur.execute("SELECT * FROM articles_meta WHERE uid = ?", (article.id,))
        meta = cur.fetchone()
        formatted = int(str(meta[2])[:-3])  # Chop off the last three digits
        date = datetime.datetime.fromtimestamp(formatted, datetime.UTC)

        # Add the article's metadata
        article.date = date.strftime("%a, %B %d %Y")
        article.url = meta[3]
        article.publication = meta[5]

        # Send the article to the client
        emit("status", "Sending article")
        emit("new_article", article.serializable())

        emit("status", "Got metadata, starting generation...")

        # summarize_article(article)


if __name__ == "__main__":
    logging
    socketio.run(app)
