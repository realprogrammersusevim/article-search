# News Article Searcher

Searches a news article database from the tool Extemp Genie using generated
embeddings and BM-25 keyword search.

## Usage

Copy the database to `news.db` and then generate the embeddings by running
`./embeddings`. Depending on how powerful your computer is this can take quite
some time. Running the web server with
`flask --app main run --host=0.0.0.0 --port 5001 --debug` will start the server
on port 5001 which you can visit by opening http://127.0.0.1:5001/ in your web
browser. Searching your query with the search bar at the top will return the
best matching articles.
