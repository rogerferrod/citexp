import re
import json


class Snippet:
    """
    Class representing a Snippet as represented in the graph

    Attributes:
        cited_articles_id: list of id of cited article
        section: section containing the snippet
        sentence: sentence containing the snippet

    Methods:
        to_csv()
            return a string in csv format representing the snippet

        text_normalize(text)
            sanitize text before being imported into the database

    """

    def __init__(self, cited_articles_id, section, sentence):
        self.cited_articles_id = cited_articles_id  # [id]
        self.section = section
        self.sentence = sentence

    def to_csv(self):
        sentence = self.text_normalize(self.sentence)
        return '"' + self.section + '";' + sentence

    @staticmethod
    def text_normalize(text):
        text = text.replace('"', "'")
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub('-\n', '', text)
        text = re.sub('\n', ' ', text)
        text = re.sub(';', ',', text)
        return json.dumps(text)
