import re
import json


class Article:
    """
    Class representing an article as represented in the graph

    Attributes:
        id: alphanumeric id (volume + article id)
        title: article's title
        authors: article's authors
        year: publication's year
        book: publication's source
        address: publication's location
        month: month of publication

    Methods:
        to_csv()
            return a string in csv format representing the article

        text_normalize(text)
            sanitize text before being imported into the database

    """

    def __init__(self, id, title, authors, year, book, address, month):
        self.id = id
        self.title = title
        self.authors = authors
        self.year = year
        self.book = book
        self.address = address
        self.month = month

    def to_csv(self):
        str_author = str(self.authors)
        str_author = str_author.replace('[', '')
        str_author = str_author.replace(']', '')
        str_author = str_author.replace("'", '')
        str_author = self.text_normalize(str_author)
        title = self.text_normalize(self.title)
        book = self.text_normalize(self.book)
        address = self.text_normalize(self.address)
        return '"' + str(self.id) + '";' + title + ';' + str_author + ';"' + self.year \
               + '"' + ';' + book + ';' + address + ';"' + self.month + '"'

    @staticmethod
    def text_normalize(text):
        text = text.replace('"', "'")
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub('-\n', '', text)
        text = re.sub('\n', ' ', text)
        return json.dumps(text)

    def __str__(self) -> str:
        return str(self.id) + " " + self.title + " " + str(self.authors) + " " + self.year
