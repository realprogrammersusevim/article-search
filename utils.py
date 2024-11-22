import datetime


class Article:
    def __init__(self, title, body, id):
        self.title = title
        self.body = body
        self.id = id
        self.embedding = None
        self.date = None
        self.url = None
        self.publication = None
        self.score = None

    def serializable(self):
        to_serial = self
        # Empty the non-serializable variables
        to_serial.embedding = None
        to_serial.similarity = None
        return to_serial.__dict__

    def add_metadata(self, date, url, publication):
        date = int(str(date)[:-3])  # Chop off the last three digits
        date = datetime.datetime.fromtimestamp(date, datetime.UTC)
        self.date = date.strftime("%a, %B %d %Y")
        self.url = url
        self.publication = publication
