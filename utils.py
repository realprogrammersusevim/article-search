import os
from typing import List
from openai import OpenAI
import markdown
from flask_socketio import emit

import tantivy


# Create the tantivy index
def create_index():
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("title", stored=True)
    schema_builder.add_text_field("body", stored=True)
    schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
    schema = schema_builder.build()

    index = tantivy.Index(schema, path=os.getcwd() + "/index")
    return index


class Article:
    def __init__(self, title, body, id):
        self.title = title
        self.body = body
        self.id = id
        self.embedding = None
        self.similarity = None
        self.date = None
        self.url = None
        self.publication = None

    def serializable(self):
        to_serial = self
        # Empty the non-serializable variables
        to_serial.embedding = None
        to_serial.similarity = None
        return to_serial.__dict__


def summarize_article(article: Article):
    client = OpenAI(api_key="lol", base_url="http://localhost:11434/v1")
    prefill = (
        "#### Main Thesis\n\n"  # Prefill the LLM to get it started on the right track
    )
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
                "content": f"Use the following format to provide a summary of the news article included below and don't deviate from the format.\n\n#### Main thesis\n\nPut the main thesis of the news article here.\n\n#### Key facts\n\n* First fact in the article, with the source in parenthesis if the article cites another source\n* Second fact, again with the source if any\n* And so on\n\nBEGIN NEWS ARTICLE\n{article['article']}\nEND NEWS ARTICLE",
            },
            {"role": "assistant", "content": prefill},
        ],
    )
    text = prefill
    for chunk in stream:
        text += chunk.choices[0].delta.content or ""
        html = markdown.markdown(text)  # Convert the markdown to HTML
        emit(
            "token",
            {
                "id": article.id,
                "text": html,
            },
        )
