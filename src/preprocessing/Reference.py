class Reference:
    """
    Class representing the reference of citation

    Attributes:
        id: numerical and progressive id
        article: id of the cited article
        marker: string used to cite
        freq: number of times it has been cited in the text
        autocit: True if the authors of the cited paper is the same of the citing paper
        title: title of cited article
        authors: authors of cited article
        date: year of cited article


    """

    def __init__(self, id):
        self.id = id
        self.article = 'unknown'
        self.marker = None
        self.freq = 0
        self.autocit = False
        self.title = None
        self.authors = []
        self.date = None

    def __str__(self) -> str:
        return str(self.id) + self.marker + str(self.freq) + str(self.autocit) + self.title + str(
            self.authors) + str(self.date)
