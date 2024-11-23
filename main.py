import datetime
import sqlite3
from contextlib import asynccontextmanager

import sentence_transformers
import sqlite_vec
import torch
from sqlite_vec import serialize_float32
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from utils import Article


async def hello_world(request):
    return templates.TemplateResponse(request, "main.html")


async def handle_search(request):
    search = request.query_params["q"]
    if search is None:
        return "No search query provided", 400

    search_embed = app.state.model.encode(search)  # Embed the search query

    # Get the articles using the embeddings
    cur = db.execute(
        "SELECT vec_articles.uid, distance FROM vec_articles WHERE embedding MATCH ? ORDER BY distance LIMIT 10",
        (serialize_float32(search_embed),),
    )
    embed_articles = cur.fetchall()
    distances = [row[1] for row in embed_articles]
    if distances:
        min_distance = min(distances)
        max_distance = max(distances)

        # Step 2: Normalize the distances to scores
        if min_distance != max_distance:  # Avoid division by zero
            normalized_results = [
                (uid, 1 - (distance - min_distance) / (max_distance - min_distance))
                for uid, distance in embed_articles
            ]
        else:  # If all distances are equal, assign a score of 1.0 to all
            normalized_results = [(uid, 1.0) for uid, distance in embed_articles]

        embed_articles = normalized_results

    # Get the articles using BM25
    cur.execute(
        """
    WITH scored_articles AS (
        SELECT
            uid,
            bm25(articles_fts, 10.0, 5.0) AS score
        FROM articles_fts
        WHERE articles_fts MATCH ?
        ORDER BY score
        LIMIT 10
    )
    SELECT uid, score FROM scored_articles;
    """,
        (search,),
    )
    bm25_articles = cur.fetchall()

    # Scale the bm25 scores to be between 0 and 1
    scores = [row[1] for row in bm25_articles]
    if scores:
        min_score = min(scores)
        max_score = max(scores)

        # Step 3: Normalize the scores
        if min_score != max_score:  # Avoid division by zero
            normalized_results = [
                (uid, (score - min_score) / (max_score - min_score))
                for uid, score in bm25_articles
            ]
        else:  # If all scores are the same, assign 1.0 to all
            normalized_results = [(uid, 1.0) for uid, score in bm25_articles]

        bm25_articles = normalized_results

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

    combined_articles += bm25_articles
    combined_articles.sort(key=lambda x: x[1], reverse=True)

    to_return = []
    for article in combined_articles:
        cur = db.execute(
            "SELECT uid, title, body, date, url, publication FROM articles WHERE uid = ?",
            (article[0],),
        )
        full_article = cur.fetchone()
        art = Article(full_article[1], full_article[2], full_article[0])
        art.add_metadata(full_article[3], full_article[4], full_article[5])
        art.score = article[1]

        to_return.append(art)

    return templates.TemplateResponse(
        request, "results.html", {"search_results": to_return}
    )


def article(request):
    id = request.path_params["id"]
    cur = db.execute(
        "SELECT uid, title, body, date, url, publication FROM articles WHERE uid = ?",
        (id,),
    )
    article = cur.fetchone()

    if article is None:
        return "Article not found", 404

    article_obj = Article(article[1], article[2], article[0])
    article_obj.add_metadata(article[3], article[4], article[5])

    return templates.TemplateResponse(request, "article.html", {"article": article_obj})


@asynccontextmanager
async def lifespan(app):
    print("Startup")
    # Set up the embedding model
    torch.device("mps")
    model = sentence_transformers.SentenceTransformer(
        "Snowflake/snowflake-arctic-embed-m"
    )
    app.state.model = model

    global db
    db = sqlite3.connect("file:search.db?mode=ro", uri=True, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    yield

    print("Shutdown")
    # For some reason this errors
    # db.disconnect()


templates = Jinja2Templates(directory="templates")
routes = [
    Route("/", hello_world),
    Route("/search", handle_search, methods=["GET"]),
    Route("/article/{id}", article),
    Mount("/static", app=StaticFiles(directory="static"), name="static"),
]
app = Starlette(debug=True, routes=routes, lifespan=lifespan)
