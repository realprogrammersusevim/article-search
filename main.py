import datetime
import sqlite3

import sentence_transformers
import sqlite_vec
import torch
from flask import Flask, render_template, request
from sqlite_vec import serialize_float32

from utils import Article

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("main.html")


@app.route("/search", methods=["GET"])
def handle_search():
    search = request.args.get("q")
    if search is None:
        return "No search query provided", 400
    # Set up the embedding model
    torch.device("mps")
    model = sentence_transformers.SentenceTransformer(
        "Snowflake/snowflake-arctic-embed-m"
    )

    search_embed = model.encode(search)  # Embed the search query

    conn = sqlite3.connect("search.db")  # Connect to the database
    cur = conn.cursor()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Get the articles using the embeddings
    cur.execute(
        "SELECT vec_articles.uid, distance FROM vec_articles WHERE embedding MATCH ? ORDER BY distance LIMIT 10",
        (serialize_float32(search_embed),),
    )
    embed_articles = cur.fetchall()

    # Scale the embedding distances to be between 0 and 1
    min_distance = embed_articles[0][1]
    max_distance = embed_articles[-1][1]
    for i in range(len(embed_articles)):
        score = (embed_articles[i][1] - min_distance) / (max_distance - min_distance)
        embed_articles[i] = (
            embed_articles[i][0],
            score,
        )
        print(score)

    # Get the articles using BM25
    cur.execute(
        "SELECT uid, bm25(articles_fts, 10.0, 5.0) FROM articles_fts WHERE articles_fts MATCH ? ORDER BY bm25(articles_fts, 10.0, 5.0) LIMIT 10",
        (search,),
    )
    bm25_articles = cur.fetchall()

    # Scale the bm25 scores to be between 0 and 1
    min_score = bm25_articles[0][1]
    max_score = bm25_articles[-1][1]
    for i in range(len(bm25_articles)):
        score = (bm25_articles[i][1] - min_score) / (max_score - min_score)
        bm25_articles[i] = (bm25_articles[i][0], score)
        print(score)

    # Combine the two lists of articles and sort them by the combined score
    combined_articles = []
    for article in embed_articles:
        uid = article[0]
        score = article[1]
        for bm25_article in bm25_articles:
            if bm25_article[0] == uid:
                score += bm25_article[1]
                bm25_articles.remove(bm25_article)
                break

        combined_articles.append((uid, score))

    combined_articles.sort(key=lambda x: x[1], reverse=True)

    to_return = []
    for article in combined_articles:
        full_article = cur.execute(
            "SELECT uid, title, body, date, url, publication FROM articles WHERE uid = ?",
            (article[0],),
        ).fetchone()
        art = Article(full_article[1], full_article[2], full_article[0])
        art.date = full_article[3]
        art.url = full_article[4]
        art.publication = full_article[5]

        to_return.append(art)

    return render_template("results.html", search_results=to_return)


@app.route("/article/<id>")
def article(id):
    conn = sqlite3.connect("search.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT uid, title, body, date, url, publication FROM articles WHERE uid = ?",
        (id,),
    )
    article = cur.fetchone()

    if article is None:
        return "Article not found", 404

    article_obj = Article(article[1], article[2], article[0])
    formatted = int(str(article[3])[:-3])  # Chop off the last three digits
    date = datetime.datetime.fromtimestamp(formatted, datetime.UTC)
    article_obj.date = date.strftime("%a, %B %d %Y")

    article_obj.url = article[4]
    article_obj.publication = article[5]
    return render_template("results.html", article=article_obj)


if __name__ == "__main__":
    import webbrowser

    webbrowser.open("http://localhost:5000/")
