import re
import time
from pathlib import Path
from lxml import etree as Exml
from optparse import OptionParser
import json

import logging
import sys

from Article import Article
from Snippet import Snippet
from Note import Note

global options
global highly_related


def clusters_extraction(file):
    """ Extracts clusters from clustering file """

    try:
        tree = Exml.parse(file)
        clusters = tree.getroot().findall("./cluster")
    except Exception as e:
        raise NameError('XML parse error ' + str(e))

    cluster_dict = {}  # cluster : [snippet_id]
    try:
        for cluster in clusters:
            label = cluster.attrib['label']
            snippets = cluster.findall('./snippet')
            cluster_dict[label] = list(map(lambda x: x.text, snippets))
    except Exception as e:
        logging.error('clusters_extraction exception ' + str(e))

    return cluster_dict


def metadata_extraction(path):
    """ Extracts metadata """

    # dtd
    dtd = '<!DOCTYPE volume ['
    dtd_file = open(path + "/ACL_XML_Metadata/100315-acl.dtd", "r")
    dtd += dtd_file.read()
    dtd_file.close()
    dtd += '<!ENTITY rsquo "&#8249;">'
    dtd += '<!ENTITY lsquo "&#8248;">'
    dtd += ']>'

    meta_path = Path(path + "/ACL_XML_Metadata")
    recursive = './*/*/*.xml'
    files_metadata = list(meta_path.glob(recursive))
    metadata = {}
    for file in files_metadata:
        try:
            # add dtd
            data = file.read_text(encoding='utf-8')
            splits = data.split('\n', 1)
            if splits[0].startswith('<?'):
                data = splits[0] + dtd + splits[1]
            else:
                data = '<?xml version="1.0" encoding="UTF-8"?>' + dtd + data

            # metadata extraction
            metadata = read_metadata(data, metadata)
        except Exception as e:
            logging.error('metadataException in ' + str(file) + " - " + str(e))
    return metadata


def read_metadata(data, metadata):
    """ Reads and parses single metadata file """
    try:
        parser = Exml.XMLParser(encoding='utf-8', remove_comments=True, resolve_entities=True)
        root = Exml.fromstring(data.encode('utf-8'), parser=parser)
        volume = root.attrib['id']

        papers = root.findall("./paper")
        for paper in papers:
            try:
                key = volume + '-' + paper.attrib['id']
                title = paper.find('title')
                title = title.text if title is not None and title.text is not None else ''
                authors = []
                for author in paper.findall('author'):
                    first = author.find('first')
                    if first is not None:
                        last = author.find('last')
                        firstname = first.text if first.text is not None else ''
                        lastname = last.text if last is not None and last.text is not None else ''
                        authors.append(firstname + ' ' + lastname)
                    else:
                        if author is not None and author.text is not None:
                            authors.append(author.text)

                year = paper.find('year')
                year = year.text if year is not None and year.text is not None else ''
                book = paper.find('booktitle')
                if book is not None and book.text is not None:
                    book = book.text
                else:
                    book = paper.find('publisher')
                    book = book.text if book is not None and book.text is not None else ''
                address = paper.find('address')
                address = address.text if address is not None and address.text is not None else ''
                month = paper.find('month')
                month = month.text if month is not None and month.text is not None else ''

                metadata[key] = Article(key, title, authors, year, book, address, month)
            except Exception as e:
                logging.error('article metadata ignored ' + str(e))
    except Exception as e:
        raise NameError('volume metadata ignored' + str(e))

    return metadata


def snippets_extraction(file):
    """ Extracts snippets from preprocessing file"""

    try:
        tree = Exml.parse(file)
        articles = tree.getroot().findall("./article")
    except Exception as e:
        raise NameError('XML parse error ' + str(e))

    snippet_dict = {}  # snippet_id : [cited_article_id]
    try:
        for article in articles:
            id = article.attrib['id']
            snippets = article.findall('./snippets/snippet')
            references = article.findall('./references/reference')
            count = 0
            for snippet in snippets:
                citations = snippet.findall('./citations/citation')
                references_id = list(map(lambda x: int(x.attrib['ref']), citations))
                cited_articles = list(filter(lambda x: int(x.attrib['id']) in references_id, references))
                cited_articles = list(map(lambda x: x.attrib['article'], cited_articles))
                cited_articles = list(filter(lambda x: x != 'unknown', cited_articles))
                section = snippet.attrib['section']
                sentence = snippet.find('sentence')
                sentence = sentence.text if sentence is not None and sentence.text is not None else ''
                citations = snippet.findall("./citations/citation")

                for c in citations:
                    sentence = sentence.replace('citation_ref=' + c.attrib['ref'], '55this_citation96')
                sentence = re.sub('citation_ref=[0-9]+', '55other_citation96', sentence)

                snippet_dict[id + '.' + str(count)] = Snippet(cited_articles, section, sentence)
                count += 1
    except Exception as e:
        logging.error('snippets_extraction exception ' + str(e))

    return snippet_dict


def import_json(file):
    """ Reads classification json file """

    json_data = open(file, 'r').read()
    classes = json.loads(json_data)
    classes = dict(map(lambda x: (int(x[0]), x[1]), classes.items()))

    return classes


def get_statistics(metadata, snippet_dict):
    """ Get some statistics """

    count_articles = len(metadata.values())
    count_no_title = 0
    count_no_author = 0
    count_no_year = 0
    count_know_references = 0
    for article in metadata.values():
        if len(article.title) == 0:
            count_no_title += 1
        if len(article.authors) == 0:
            count_no_author += 1
        if len(article.year) == 0:
            count_no_year += 1

    for snippet in snippet_dict.values():
        count_know_references += len(snippet.cited_articles_id)

    logging.info('metadata ' + str(count_articles))
    percent = (count_no_title / count_articles) * 100
    logging.info('no title ' + str(count_no_title) + ' (' + "%0.2f" % percent + '%)')
    percent = (count_no_author / count_articles) * 100
    logging.info('no author ' + str(count_no_author) + ' (' + "%0.2f" % percent + '%)')
    percent = (count_no_year / count_articles) * 100
    logging.info('no year ' + str(count_no_year) + ' (' + "%0.2f" % percent + '%)')
    logging.info('known references ' + str(count_know_references))


def main():
    global options
    global highly_related

    output_log = options.output + "/graph.log"
    logging.basicConfig(filename=output_log, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    articles_csv = open(options.output + '/articles.csv', 'w', encoding='utf-8')
    citations_csv = open(options.output + '/citations.csv', 'w', encoding='utf-8')
    objects_csv = open(options.output + '/objects.csv', 'w', encoding='utf-8')
    purposes_csv = open(options.output + '/purposes.csv', 'w', encoding='utf-8')

    # get data
    logging.info('loading...')
    t0 = time.time()
    cluster_dict = clusters_extraction(options.cluster_input)
    metadata = metadata_extraction(options.acl_path)
    snippet_dict = snippets_extraction(options.preprocessing_input)
    class_dict = import_json(options.json_input)
    logging.info("done in %0.3fs" % (time.time() - t0))

    # statistics
    get_statistics(metadata, snippet_dict)

    # articles csv
    logging.info('creating articles csv...')
    t0 = time.time()
    articles_csv.write('"id";"title";"authors";"year";"book";"address";"month"' + '\n')
    for article in metadata.values():
        if article.title is not None:
            articles_csv.write(article.to_csv() + '\n')

    logging.info("done in %0.3fs" % (time.time() - t0))

    # citations csv
    logging.info('creating citations csv...')
    purpose = []

    t0 = time.time()
    citations_csv.write('"source";"target";"class";"snippet";"section";"sentence"' + '\n')
    for snippet_id, snippet in snippet_dict.items():
        if len(snippet.cited_articles_id) > 0:  # != unknown
            article_id = snippet_id.split('.')[0]
            cluster = None
            for c, s_list in cluster_dict.items():
                if snippet_id in s_list:
                    cluster = c
                    break

            if cluster is not None:
                label = class_dict[int(cluster)] if int(cluster) in class_dict.keys() else 'related-to'
                for cited_id in snippet.cited_articles_id:
                    str_label = label if label not in highly_related else 'highly-related'
                    citations_csv.write(
                        '"' + article_id + '";"' + cited_id + '";"' + str_label + '";"' + snippet_id + '";' + snippet.to_csv() + '\n')

                    if label in highly_related:
                        purpose += [Note(label, snippet.sentence, cited_id)]
                        # TODO turn it into a set with a similarity metric

    logging.info("done in %0.3fs" % (time.time() - t0))

    # object csv
    logging.info('creating object csv...')
    t0 = time.time()
    objects_csv.write('"id";"object";"sentence"' + '\n')
    for i in range(len(purpose)):
        objects_csv.write('"' + str(i) + '";' + purpose[i].note + ';' + purpose[i].sentence + '\n')
    logging.info("done in %0.3fs" % (time.time() - t0))

    # purpose csv
    logging.info('creating purpose csv...')
    t0 = time.time()
    purposes_csv.write('"source";"target";"purpose"' + '\n')
    for i in range(len(purpose)):
        purposes_csv.write('"' + purpose[i].source + '";"' + str(i) + '";"' + purpose[i].purpose + '"' + '\n')
    logging.info("done in %0.3fs" % (time.time() - t0))


if __name__ == "__main__":
    print("Graph")

    highly_related = ['approach', 'present', 'proposed by', 'introduce', 'use', 'report', 'method']

    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-c", "--clusters", help='clusters input file', action="store", type="string",
                      dest="cluster_input")
    parser.add_option("-a", "--aclpath", help='ACL Anthology path', action="store", type="string", dest="acl_path")
    parser.add_option("-p", "--preprocessing", help='preprocessing input file', action="store", type="string",
                      dest="preprocessing_input")
    parser.add_option("-j", "--json", help='json class input file', action="store", type="string",
                      dest="json_input")
    parser.add_option("-o", "--output", help='output folder', action="store", type="string", dest="output")

    (options, args) = parser.parse_args()

    if options.cluster_input is None or options.output is None or options.acl_path is None \
            or options.preprocessing_input is None or options.json_input is None:
        print("Missing parameters")
        sys.exit(2)

    main()
