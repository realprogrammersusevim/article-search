# News Article Searcher

Hybrid search a news article database from the tool Extemp Genie using
embeddings and BM-25 keyword search.

## Installation

Clone the repo, install the requirements, and get the database.

```bash
git clone https://github.com/realprogrammersusevim/article-search
cd article-search
python3 -m venv .venv # Or $ uv venv
source .venv/bin/activate
pip install -r requirements.txt
```

Now the code and dependencies are installed let's get the Extemp Genie database.
Go to your web browsers folder for local storage. For example, on macOS with
Google Chrome that's
`~/Library/Application Support/Google/Chrome/Default/File System/`. Since the
article database is generally several gigabytes the easiest way from here is to
use a disk explorer like diskonaut to go to the largest file in the
`File System/` directory. Copy the largest file you find to
`article-search/news.db`

## Processing Articles

To perform fast searches we need a new database of processed articles. Run the
`embeddings` script to get all the articles from the `news.db` database and
write them to the new database with embeddings. The script will process the
articles in chunks of 20 batches at a time (each batch is 32 articles) and write
each chunk to the database, meaning that you can safely stop the script without
losing progress. The whole process of generating embeddings for the articles
takes several hours for six months of articles depending on how beefy your
computer is.

## Usage

Run the server with `uvicorn main:app` and open https://localhost:8000 in your
web browser. Entering a search will run a hybrid query that uses SQLite's Full
Text Search and BM25 ranking as well as semantic search using the embeddings we
generated.

## TODO

- [ ] Re-add summarization using the Llama 3b models
- [ ] Save articles to sidebar to read later
- [ ] Test other forms of result synthesis besides adding scaled scores
