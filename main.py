import datetime
import pickle
import sqlite3

import numpy as np
import sentence_transformers
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from openai import OpenAI

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
    emit("status", "Loaded model")
    # Load the data
    with open("dump.pickle", "rb") as f:
        everything = pickle.load(f)

    emit("status", "Loaded data")

    # Embed the search query
    search_embed = model.encode(search)
    model = None  # Free up memory and stop the "Oh no you forked the process" error
    emit("status", "Embedded search query")

    # Calculate the similarity between the search query and each article
    for i, art in enumerate(everything):
        embed = art["embedding"]
        similarity = np.dot(embed, search_embed) / (
            np.linalg.norm(embed) * np.linalg.norm(search_embed)
        )
        art["similarity"] = similarity

    emit("status", "Calculated similarities")

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
        print(curr_article["id"])
        emit("status", f"Processing article {curr_article['id']}")

        cur.execute("SELECT * FROM articles_meta WHERE uid = ?", (curr_article["id"],))
        meta = cur.fetchone()
        formatted = int(str(meta[2])[:-3])
        date = datetime.datetime.fromtimestamp(formatted, datetime.UTC)

        # Add the article's metadata
        curr_article["date"] = date.strftime("%a, %B %d %Y")
        curr_article["url"] = meta[3]
        curr_article["publication"] = meta[5]

        # Change article to be JSON serializable
        curr_article.pop("embedding", None)
        curr_article.pop("similarity", None)

        # Send the article to the client
        print("Sending article")
        emit("status", "Sending article")
        emit("new_article", curr_article)

        print("Got metadata, starting generation...")
        emit("status", "Got metadata, starting generation...")

        client = OpenAI(api_key="lol", base_url="http://localhost:11434/v1")
        stream = client.chat.completions.create(
            model="llama3",
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": "You are a seasoned writer and summarizer.",
                },
                {
                    "role": "user",
                    "content": f"Use the following format to provide a summary of the news article included below and don't deviate from the format.\n\n<h4>Main thesis</h4><p>Put the main thesis of the news article here.<h4>Key facts</h4></p><ul><li>First fact in the article, with the source in parenthesis if the article cites another source</li><li>Second fact, again with the source if any</li><li>And so on</li></ul>\n\nBEGIN NEWS ARTICLE\n{article['article']}\nEND NEWS ARTICLE",
                },
                {"role": "assistant", "content": "<h4>Main thesis</h4>"},
            ],
        )
        text = "<h4>Main thesis</h4>"  # We've already pre-filled the AI response
        for chunk in stream:
            text += chunk.choices[0].delta.content or ""
            emit(
                "token",
                {
                    "id": curr_article["id"],
                    "text": text,
                },
            )


if __name__ == "__main__":
    socketio.run(app)
