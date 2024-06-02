import os

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
        return lambda o: o.__dict__
