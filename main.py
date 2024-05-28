import argparse
import datetime
import pickle
import sqlite3

import numpy as np
import requests
import sentence_transformers
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def hello_world():
    return render_template("index.html")


@socketio.on("search")
def handle_search(search):
    print(search)

    # Set up the embedding model
    model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2", device="mps")
    # Load the data
    with open("dump.pickle", "rb") as f:
        everything = pickle.load(f)

    # Embed the search query
    search_embed = model.encode(search)

    # Calculate the similarity between the search query and each article
    for i, art in enumerate(everything):
        embed = art["embedding"]
        similarity = np.dot(embed, search_embed) / (
            np.linalg.norm(embed) * np.linalg.norm(search_embed)
        )
        art["similarity"] = similarity

    # Connect to the database
    conn = sqlite3.connect("news.db")
    cur = conn.cursor()

    # Sort the articles by similarity
    sorted_art = sorted(everything, key=lambda x: x["similarity"])

    # Get the top 10 articles
    articles = sorted_art[-10:]

    for article in articles:
        # The article we're working with
        curr_article = article

        cur.execute("SELECT * FROM articles_meta WHERE uid = ?", (curr_article["id"],))
        meta = cur.fetchone()
        formatted = int(str(meta[2])[:-3])
        date = datetime.datetime.fromtimestamp(formatted, datetime.UTC)
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "stream": False,
                "prompt": f"Use the following format to provide a summary of the news article included below and don't deviate from the format.\n\n<h4>Main thesis</h4><p>Put the main thesis of the news article here.<h4>Key facts</h4></p><ul><li>First fact in the article, with the source in parenthesis if the article cites another source</li><li>Second fact, again with the source if any</li><li>And so on</li></ul>\n\nBEGIN NEWS ARTICLE\n{article['article']}\nEND NEWS ARTICLE",
            },
        ).json()

        # Add the article's metadata
        curr_article["date"] = date.strftime("%a, %B %d %Y")
        curr_article["summary"] = resp["text"]
        curr_article["url"] = meta[3]
        curr_article["publication"] = meta[5]

        # Send the article to the client
        emit("new_article", curr_article)


if __name__ == "__main__":
    socketio.run(app)
