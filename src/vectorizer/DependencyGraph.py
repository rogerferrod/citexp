import re


class DependencyGraph:
    """
    Class representing the dependency graph of a sentence.
    the syntatic parser supplies a list of edges labeled with the dependency
        example: 2 -> 6[label='nmod']
    and a list of nodes labeled with the corresponding word in the sentence
        example: 2[label='2(some)']
                 6[label='6(toys)']

    Attributes:
        graph: adjacent list graph, edges labeled with dependencies
        dict: dictionary {node : word of sentence}

    Methods:
        from_dot(dot_notation)
            initialize graph from parser's tree supplied in dot notation

        get_bidirectional_adj(value)
            get a set of bidirectional adjacent nodes of node 'value'

        get_index(value)
            get a list of nodes corresponding to the word 'value'

    """

    def __init__(self):
        self.graph = {}
        self.dict = {}

    def from_dot(self, dot_notation):
        lines = dot_notation.split('\n')
        lines = lines[4:-1]
        for line in lines:
            splits = re.split('\[label="', line, maxsplit=2)
            splits[0] = splits[0].strip()
            splits[1] = str(splits[1][:-2]).strip()
            if re.match('^[0-9]+$', splits[0]):
                labels = re.findall('\(([^)]+)\)', splits[1])
                self.dict[int(splits[0])] = labels[0]
                self.graph[int(splits[0])] = []
            else:
                label = splits[1].strip()
                edge = splits[0].split('->')
                self.graph[int(edge[0])].append((int(edge[1]), label))

    def get_bidirectional_adj(self, value):
        directional = self.graph[value]  # directional adj
        inverse_keys = []
        for k, v in self.graph.items():
            for e in v:
                if e[0] == value and e[1] != 'ROOT':
                    inverse_keys.append(k)
                    break  # internal loop

        adjs = set(directional)
        inverse = set()
        for k in inverse_keys:
            edges = self.graph[k]
            for e in edges:
                if e[0] == value:
                    inverse.add((k, e[1]))

        adjs = adjs.union(inverse)

        return adjs

    def get_index(self, value):
        keys = []
        for k, v in self.dict.items():
            if v == value:
                keys.append(k)
        return keys
