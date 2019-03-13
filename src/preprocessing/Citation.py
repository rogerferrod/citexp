class Citation:
    """
    Class representing single citation as extracted from parscit

    Attributes:
        id: numerical and progressive id
        ref: reference id
        authors: authors of citated article
        context: context as extracted from parscit (1220 characters)
        sentence: sentence containing the citation
        raw_str: citing string as reported in original text
        autocit: True if the authors of the cited paper is the same of the citing paper
        section: section containing the citation
        position: position of citation (in character) relative to the entire text


    """

    def __init__(self, id, ref):
        self.id = id
        self.ref = ref
        self.authors = []
        self.context = None
        self.sentence = None
        self.raw_str = None
        self.autocit = False
        self.section = None
        self.position = 0

    def __str__(self) -> str:
        return str(self.id) + str(self.ref) + str(
            self.authors) + self.context + self.sentence + self.raw_str + str(self.autocit) + self.section + str(
            self.position)
