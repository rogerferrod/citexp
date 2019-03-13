from lxml import etree as Exml
from optparse import OptionParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle
import logging

import re
import time
import sys

from DependencyGraph import DependencyGraph


def snippets_extraction(file):
    """ Extracts snippets from preprocessing file.
    Returns a dictionary {snippet_id : sentence}

    """

    try:
        tree = Exml.parse(file)
        articles = tree.getroot().findall("./article")
    except Exception as e:
        raise NameError('XML parse error ' + str(e))

    snippet_dict = {}
    try:
        for article in articles:
            snippets = article.findall('./snippets/snippet')
            count = 0
            for snippet in snippets:
                sent = snippet.find("./sentence")
                sent = sent.text
                citations = snippet.findall("./citations/citation")

                for c in citations:
                    sent = sent.replace('citation_ref=' + c.attrib['ref'], '55this_citation96')
                sent = re.sub('citation_ref=[0-9]+', '55other_citation96', sent)

                snippet_dict[article.attrib['id'] + '.' + str(count)] = sent
                count += 1
    except Exception as e:
        logging.error('snippets_extraction exception ' + str(e))

    return snippet_dict


def tokenize(sentence):
    """ tf-idf tokenize overriding.
     Override the scikit-learn tokenize function

     """

    global options

    dependency = ['nsubj', 'csubj', 'nmod', 'advcl', 'dobj']
    placeholders = ['55this_citation96', '55other_citation96']

    # pre-processing
    sentence = sentence.replace('(', '')
    sentence = sentence.replace(')', '')
    sentence = sentence.replace('[', '')
    sentence = sentence.replace(']', '')

    # Stanford CoreNLP parser
    try:
        dependency_parser = CoreNLPDependencyParser(url=options.url)
        result = dependency_parser.raw_parse(sentence)
        dep = next(result)
    except Exception as e:
        raise NameError(e)

    try:
        # dependency graph
        graph = DependencyGraph()
        graph.from_dot(dep.to_dot())

        tokens = []
        cit_indices = graph.get_index('55this_citation96')
        for cit_index in cit_indices:
            first_level = graph.get_bidirectional_adj(cit_index)

            if cit_index in list(graph.dict.keys())[-2:]:  # citation at the end of sentence
                first_level = list(filter(lambda x: x[1] in dependency, first_level))

            if len(first_level) != 0:
                first_level = list(map(lambda x: x[0], first_level))
                triggers = []  # words syntactically linked to the citation
                triggers += first_level
                frontier = first_level

                # trigger-words
                for i in range(1, options.depth):
                    new_frontier = []
                    for n in frontier:
                        discovered = graph.get_bidirectional_adj(n)
                        discovered = list(filter(lambda x: x[0] not in cit_indices and x[1] in dependency, discovered))
                        discovered = list(map(lambda x: x[0], discovered))
                        new_frontier += discovered
                    frontier = new_frontier
                    triggers += new_frontier

                # filter trigger
                triggers = list(filter(
                    lambda x: graph.dict[x] not in placeholders
                              and graph.dict[x] != 'None'
                              and graph.dict[x] not in stopwords.words('english')
                              and re.match('[a-zA-Z]{2,}', graph.dict[x]), triggers))

                # ngram
                for x in triggers:
                    ngram = []  # all words from trigger to citation
                    i = x
                    if i == cit_index:
                        ngram.append(graph.dict[i])
                    if i < cit_index:
                        while i < cit_index:
                            word = graph.dict[i]
                            if re.match('[a-zA-Z]', word) or word in placeholders:
                                ngram.append(word)
                            i += 1
                    else:
                        while i > cit_index:
                            word = graph.dict[i]
                            if re.match('[a-zA-Z]', word) or word in placeholders:
                                ngram.append(word)
                            i -= 1
                        ngram.reverse()

                    if len(ngram) > 0:
                        tokens.append(tuple(ngram))

        # stemmer
        stemmer = PorterStemmer()
        tokens = list(map(lambda x: tuple(map(lambda y: stemmer.stem(y), x)), tokens))

    except Exception as e:
        logging.error('Tokenize exception ' + str(e))
        tokens = []

    return tokens


def vectorizer(sentences):
    """ tf-idf vectorizer and features extraction.
     Applies the tf-idf algorithm,
     returns the tf-idf matrix and the list of features used

     """

    global options

    # tf-idf vectorizer
    logging.info('tf-idf vectorizer...')
    tfidf = TfidfVectorizer(tokenizer=tokenize, max_features=10000, use_idf=True)

    t0 = time.time()
    matrix = tfidf.fit_transform(sentences)
    logging.info("done in %0.3fs" % (time.time() - t0))
    logging.info(matrix.shape)

    features = tfidf.get_feature_names()

    return matrix, features


def main():
    global options

    output_log = options.output + "/vectorizer.log"
    logging.basicConfig(filename=output_log, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # snippets
    snippet_dict = snippets_extraction(options.input)

    if options.input_limit == -1:
        snippets = list(snippet_dict.items())
    else:
        snippets = list(snippet_dict.items())[:options.input_limit]

    snippet_dict = dict(snippets)

    logging.info('num_sentences ' + str(len(snippets)))
    sentences = list(map(lambda x: x[1], snippets))

    # vectorizer
    matrix, features = vectorizer(sentences)

    # dump
    logging.info('dump...')
    sparse.save_npz(options.output + '/matrix.npz', matrix)
    with open(options.output + '/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    with open(options.output + '/snippet_dict.pkl', 'wb') as f:
        pickle.dump(snippet_dict, f)


if __name__ == "__main__":
    print("Vectorizer")

    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-i", "--input", help='preprocessing file', action="store", type="string", dest="input")
    parser.add_option("-o", "--output", help='output folder', action="store", type="string", dest="output")
    parser.add_option("-u", "--url", help='server url', action="store", type="string", dest="url")
    parser.add_option("-d", "--depth", help='graph depth', action="store", type="int", dest="depth", default=2)
    parser.add_option("-l", "--limit", help='input limit', action="store", type="int", dest="input_limit", default=-1)

    (options, args) = parser.parse_args()

    if options.input is None or options.url is None or options.output is None:
        print("Missing parameters")
        sys.exit(2)

    main()
