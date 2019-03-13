import xml.etree.ElementTree as ET


class Article:
    """
    Class representing an article as extracted from parscit with metadata

    Attributes:
        id: alphanumeric id (volume + article id)
        title: article's title
        authors: article's authors
        sections: length of each section, dictionary {section : len_text}
        input_xml: tree of parsed xml
        snippets: list of snippets
        references: list of references


    """

    def __init__(self, id, title, authors):
        self.id = id
        self.title = title
        self.authors = authors
        self.sections = {}
        self.input_xml = None
        self.snippets = []
        self.references = []

    def to_xml(self):
        article = ET.Element('article')
        article.set('id', self.id)
        title = ET.SubElement(article, 'title')
        title.text = self.title
        authors = ET.SubElement(article, 'authors')
        for a in self.authors:
            author = ET.SubElement(authors, 'author')
            author.text = a
        snippets = ET.SubElement(article, 'snippets')
        references = ET.SubElement(article, 'references')
        for s in self.snippets:
            cit = s.citation
            snippet = ET.SubElement(snippets, 'snippet')
            snippet.set('raw_string', cit.raw_str)
            snippet.set('num_cit', str(len(s.references)))
            snippet.set('section', cit.section)

            citations = ET.SubElement(snippet, 'citations')
            for ref in s.references:
                citation = ET.SubElement(citations, 'citation')
                citation.set('ref', str(ref))

            sentence = ET.SubElement(snippet, 'sentence')
            sentence.text = cit.sentence
            context = ET.SubElement(snippet, 'context')
            context.text = cit.context

        for r in self.references:
            reference = ET.SubElement(references, 'reference')
            reference.set('id', str(r.id))
            reference.set('article', r.article)
            reference.set('marker', r.marker)
            reference.set('freq', str(r.freq))
            reference.set('autocit', str(r.autocit))
            title = ET.SubElement(reference, 'title')
            title.text = r.title
            authors = ET.SubElement(reference, 'authors')
            for a in r.authors:
                author = ET.SubElement(authors, 'author')
                author.text = a
            date = ET.SubElement(reference, 'date')
            date.text = r.date

        tree = ET.ElementTree(article)

        return tree

    def __str__(self) -> str:
        return str(self.id) + " " + self.title + " " + str(self.authors)
