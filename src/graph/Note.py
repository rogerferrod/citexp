import re
import json
import nltk


class Note:
    """
    Class representing a semantic note linked to a citation

    Attributes:
        purpose: purpose of citation
        note: semantic note (piece of text surrounding the citation)
        sentence: sentence containing the citation
        source: id of cited article

    Methods:
        extract_object(sentence)
            extract semantic note from the sentence

        text_normalize(text)
            sanitize text before being imported into the database

    """

    def __init__(self, purpose, sentence, source):
        sentence = self.text_normalize(sentence)
        self.purpose = purpose
        self.note = self.extract_object(sentence)
        self.sentence = sentence
        self.source = source

    @staticmethod
    def extract_object(sentence):
        placeholder = '55this_citation96'
        tokens = nltk.word_tokenize(sentence)
        index = tokens.index(placeholder)
        for i in range(len(tokens)):
            if tokens[i] == placeholder and i != index:
                tokens[i] = 'CIT'
        limit = 3

        if index <= 1:
            begin = index
            end = index + limit * 2
        elif tokens[index - 1] == ',':
            begin = index
            end = index + limit * 2
        else:
            begin = index - limit
            end = index + limit

        truncated = tokens[begin if begin > 0 else 0: end + 1]
        extend = truncated.count('CIT')
        if extend > 0:
            truncated += tokens[end + 1: end + extend + 1]

        string = '[...] '
        for w in truncated:
            string += w + ' '
        string += '[...]'
        return string

    @staticmethod
    def text_normalize(text):
        text = text.replace('"', "'")
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub('-\n', '', text)
        text = re.sub('\n', ' ', text)
        text = re.sub(';', ',', text)
        return json.dumps(text)
