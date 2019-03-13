class Snippet:
    """
    Class representing a snippet

    Attributes:
        id: citation's id
        references: list of references of cited article


    """

    def __init__(self):
        self.citation = None
        self.references = []

    def __eq__(self, o: object) -> bool:
        if isinstance(self, o.__class__):
            return self.references == o.references and self.citation.sentence == o.citation.sentence
        return False

    def __str__(self) -> str:
        return "[" + str(self.references) + "] " + self.citation.sentence
