from pathlib import Path
from lxml import etree as Exml
from optparse import OptionParser
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from difflib import SequenceMatcher
import progressbar
import nltk

import re
import math
import time
import logging
import sys
import os

from Article import Article
from Citation import Citation
from Snippet import Snippet
from Reference import Reference

count_metadata_article_errors = 0
count_matches = 0


def metadata_extraction(data, metadata):
    """ Extracts metadata.
     extracts metadata from xml file 'data' and store results in 'metadata' dictionary

     """
    global count_metadata_article_errors

    try:
        parser = Exml.XMLParser(encoding='utf-8', remove_comments=True, resolve_entities=True)
        root = Exml.fromstring(data.encode('utf-8'), parser=parser)
        volume = root.attrib['id']

        papers = root.findall("./paper")
        for paper in papers:
            try:
                id = paper.attrib['id']
                key = volume + '-' + id
                title = paper.find('title')
                title = title.text if title is not None else ''
                authors = paper.findall('author')
                authors = list(map(lambda x: x.text, authors))
                authors = sorted(filter(lambda x: x is not None, authors))
                metadata[key] = (title, authors)
            except Exception as e:
                count_metadata_article_errors += 1
                logging.error('article metadata ignored ' + str(e))
    except Exception as e:
        raise NameError('volume metadata ignored' + str(e))

    return metadata


def article_process(path, metadata):
    """ Process article file.
    Analyses single article file ('path'), returns parsed articled as Article object

     """

    # get corresponding metadata
    tail = os.path.split(path)[1].split('-parscit')[0]
    if tail in metadata:
        meta = metadata[tail]
    else:
        meta = ('', [])  # article not identifiable but still useful

    # article extraction
    article = article_extraction(str(path), id=tail, title=meta[0], authors=meta[1])

    # citation extraction
    citations, references = cit_extraction(article)

    # citations refining
    citations, references = cit_refining(article, metadata, citations, references)

    # snippet extraction
    snippets = create_snippets(citations)

    article.snippets = snippets
    article.references = references.values()

    return article


def article_extraction(file, id, title, authors):
    """ Extracts article.
    Extracts article's information from parscit

     """

    article = Article(id, title, authors)
    try:
        tree = Exml.parse(file)
        article.input_xml = tree
        xml_iter = tree.getroot().findall("./algorithm[@name='SectLabel']/variant/*")
    except Exception as e:
        raise NameError('XML parse error ' + str(e))

    sections = {}
    text = ''
    header = 'header'
    index = 0

    # sections (only header)
    while index < len(xml_iter):
        child = xml_iter[index]
        if child.tag != 'sectionHeader':
            text = text + child.text
        else:
            sections[header] = len(text_normalize(text))
            break
        index += 1

    # sections (from abstract to reference)
    text = ''
    while index < len(xml_iter):
        child = xml_iter[index]
        if child.tag == 'reference':
            break
        if child.tag == 'sectionHeader':
            if 'genericHeader' in child.attrib and child.attrib['genericHeader'] != header:
                sections[header] = len(text_normalize(text))
                header = child.attrib['genericHeader']
                text = child.text
        else:
            text = text + child.text
        index += 1

    article.sections = sections

    return article


def cit_extraction(article):
    """ Extracts citations.
    Extracts citations from parscit

     """

    tree = article.input_xml
    try:
        cits = tree.getroot().findall("./algorithm[@name='ParsCit']/citationList/citation[@valid='true'][contexts]")
    except Exception:
        raise NameError('XML parse error')

    citations = []
    synonyms = {}
    references = {}

    # sentence tokenizer
    punkt_param = PunktParameters()
    abbreviation = ['e.g', 'i.e', 'al', 'seq', 'fig', 'pp', 'vol', 'n.d', 'vs', 'pag', 'ib']
    punkt_param.abbrev_types = set(abbreviation)
    sent_tokenizer = PunktSentenceTokenizer(punkt_param)

    for cit in cits:
        marker = cit.find('./marker').text
        synonyms[marker] = set()
        for context in cit.findall('./contexts/context'):
            try:
                text = context.text
                cit_str = context.attrib['citStr']
                cit_position = int(context.attrib['position'])  # position (middle) relative to the entire text
                context_position = math.ceil(len(text) / 2)  # position (middle) relative to context
                citation_id = len(citations) + 1
                citation_ref = list(synonyms.keys()).index(marker) + 1
                synonyms[marker].add(cit_str)  # add synonyms

                # placeholder
                replacement = ' citation_ref=' + str(citation_ref) + ' '
                text = text.replace(cit_str, replacement)

                # sentence extraction
                sentences = sent_tokenizer.tokenize(text)
                sentence = None
                count = 0
                for s in sentences:
                    count += len(s)
                    if context_position < count:
                        sentence = s
                        break

                if sentence is None:
                    raise NameError('cannot extract sentence')

                # append citation and reference
                title = cit.find('title')
                date = cit.find('date')

                reference = Reference(citation_ref)
                reference.marker = marker
                reference.title = title.text if title is not None else ''
                reference.authors = list(map(lambda x: x.text.strip(), cit.findall('./authors/author')))
                reference.date = date.text if date is not None else ''

                references[citation_ref] = reference

                citation = Citation(citation_id, citation_ref)
                citation.raw_str = cit_str
                citation.context = text
                citation.sentence = sentence
                citation.position = cit_position

                citations.append(citation)

            except Exception as e:
                logging.error('cit extraction ' + str(e))

    # place holder
    citations = replace_references(citations, synonyms)

    return citations, references


def cit_refining(article, metadata, citations, references):
    """ Refine informations.
    add information to citations (section, autocit, frequency, internal article reference)

    """

    global count_matches, options

    # sectionize
    sections = list(article.sections.items())
    for c in citations:
        offset = 0
        for t in sections:
            next_offset = offset + t[1] - 1
            if offset <= c.position <= next_offset:
                c.section = t[0]
                break
            offset = next_offset + 1
        if c.section is None:
            c.section = 'unknown'

    # autocit
    for ref in references.values():
        ref.autocit = set(ref.authors).issubset(set(article.authors))

    # frequency
    sentences = list({x.sentence for x in citations})
    freq = {}
    for k in references.keys():
        count = 0
        for s in sentences:
            count = count + s.count('citation_ref=' + str(k))
        freq[k] = count

    for ref in references.values():
        ref.freq = freq[ref.id]

    # internal article matches
    matches = 0
    if options.do_matches:
        for ref in references.values():
            for k, v in metadata.items():
                a1 = str(v[1])
                a2 = str(ref.authors)
                v = str(v[0]).lower()
                t = str(ref.title).lower()
                if SequenceMatcher(None, v, t).quick_ratio() > 0.92:
                    ref.article = k
                    matches += 1
                    break

    count_matches += matches

    return citations, references


def create_snippets(citations):
    """ Creates snippets.
    creates snippets from raw citations

    """

    snippets = []
    mulcits = []
    grammar = '''
        MULCIT: {<CIT>((<CC>|<,>|<.>|<;>|<:>)?<CIT>)+}
        SINGLECIT :   {<CIT>}
        '''

    for c in citations:
        try:
            mulcit_tree = []
            single_tree = []

            # part of speech tagging
            tokens = nltk.word_tokenize(c.sentence)
            pos_tag = nltk.pos_tag(tokens)  # default: english
            pos_tag = list(map(lambda x: (x[0], 'CIT') if re.match('citation_ref=[0-9]+', x[0]) else x, pos_tag))

            # parser
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(pos_tag)
            subtrees = tree.subtrees(filter=lambda x: x != tree)

            for s in subtrees:
                labels = s.label()
                if labels[0] == 'M':
                    mulcit_tree.append(s)
                else:
                    single_tree.append(s)

            # single trees
            if len(single_tree) > 0:
                snippet = Snippet()
                snippet.citation = c
                snippet.references = [c.ref]
                snippets.append(snippet)

            # mulcit trees
            if len(mulcit_tree) > 0:
                for s in mulcit_tree:
                    mulcit = list(filter(lambda x: x[1] == 'CIT', s.leaves()))
                    mulcit = list(map(lambda x: int(x[0].split('=')[1]), mulcit))
                    snippet = Snippet()
                    snippet.citation = c
                    snippet.references = mulcit
                    if snippet not in mulcits:
                        mulcits.append(snippet)

        except Exception as e:
            logging.error('cannot extract citation ' + str(e))

    return snippets + mulcits


def text_normalize(text):
    """ Parscit text processing
    redo parscit normalizing transformations on text

    """

    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('-\n', '', text)
    text = re.sub('\n', ' ', text)

    return text


def replace_references(citations, synonyms):
    """ Replaces reference with placeholder """

    keys = list(synonyms.keys())
    for ref in synonyms.keys():
        syn = synonyms[ref]
        code = keys.index(ref) + 1
        for c in citations:
            for s in syn:
                c.sentence = c.sentence.replace(s, ' citation_ref=' + str(code) + ' ')
                c.context = c.context.replace(s, ' citation_ref=' + str(code) + ' ')

    return citations


def main():
    global count_matches, statusbar, options

    output_path = options.output
    path = options.path
    begin = options.begin
    end = options.end
    xml_prefix = options.xml_prefix

    output_xml = output_path + "/preprocessing.xml"
    output_log = output_path + "/preprocessing.log"

    logging.basicConfig(filename=output_log, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    num_references = 0
    num_snippets = 0
    count_metadata_errors = 0
    count_article_errors = 0
    count_article_no_title = 0
    count_article_no_authors = 0

    out = open(output_xml, 'ab')  # append-binary
    if xml_prefix:
        out.write(b'<?xml version="1.0" encoding="UTF-8"?>')
        out.write(b'<articles>')

    # list of files to analyse
    article_path = Path(path + "/ParsCitXML")
    meta_path = Path(path + "/ACL_XML_Metadata")
    recursive = './*/*/*.xml'
    files_articles = list(article_path.glob(recursive))
    files_metadata = list(meta_path.glob(recursive))
    num_corpus_articles = len(files_articles)

    # parametric restriction
    logging.info('Articles[' + str(begin) + ':' + str(end) + ']')
    files_articles = files_articles[begin:end]
    slice_len = len(files_articles)

    # read dtd (for metadata)
    dtd = '<!DOCTYPE volume ['
    dtd_file = open(path + "/ACL_XML_Metadata/100315-acl.dtd", "r")
    dtd += dtd_file.read()
    dtd_file.close()
    dtd += '<!ENTITY rsquo "&#8249;">'
    dtd += '<!ENTITY lsquo "&#8248;">'
    dtd += ']>'

    # statusbar
    statusbar = progressbar.ProgressBar(maxval=slice_len + 1,
                                        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    # metadata
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
            metadata = metadata_extraction(data, metadata)
        except Exception as e:
            count_metadata_errors += 1
            logging.error('metadataException in ' + str(file) + " - " + str(e))

    # articles
    i = 0
    statusbar.start()
    start_time = time.time()
    for file in files_articles:
        try:
            # article extraction
            article = article_process(file, metadata)
            if article.title == '':
                count_article_no_title += 1
            if len(article.authors) == 0:
                count_article_no_authors += 1
            num_references += len(article.references)
            num_snippets += len(article.snippets)
            i += 1
            statusbar.update(i + 1)

            if options.verbose:
                logging.info(article)

            # write output
            if len(article.snippets) > 0 and len(article.references) > 0:
                try:
                    article.to_xml().write(out, encoding='utf-8')
                except Exception as e:
                    logging.error('Invalid data ' + article.id + ' - ' + str(e))
                    count_article_errors += 1
        except Exception as e:
            logging.error('Ignored ' + str(file) + ' - ' + str(e))
            count_article_errors += 1

    if xml_prefix:
        out.write(b'</articles>')

    elapsed_time = time.time() - start_time
    statusbar.finish()

    logging.info('Time elapsed ' + str(elapsed_time))
    logging.info('ACL Anthology articles ' + str(num_corpus_articles))
    percent = (slice_len / num_corpus_articles) * 100
    logging.info('tot slice ' + str(slice_len) + ' (' + "%0.2f" % percent + '%)')
    percent = 1 - len(metadata.values()) / num_corpus_articles
    logging.info(
        'metadata ignored ' + str(num_corpus_articles - len(metadata.values())) + ' (' + "%0.2f" % percent + '%)')
    percent = count_article_errors / len(files_articles) * 100
    logging.info('article ignored ' + str(count_article_errors) + ' (' + "%0.2f" % percent + '%)')
    percent = count_article_no_title / len(files_articles) * 100
    logging.info('article (no title) ' + str(count_article_no_title) + ' (' + "%0.2f" % percent + '%)')
    percent = count_article_no_authors / len(files_articles) * 100
    logging.info('article (no authors) ' + str(count_article_no_authors) + ' (' + "%0.2f" % percent + '%)')
    logging.info('snippets ' + str(num_snippets))
    logging.info('references ' + str(num_references))
    percent = count_matches / num_references * 100
    logging.info('matches ' + str(count_matches) + ' (' + "%0.2f" % percent + '%)')


if __name__ == "__main__":
    print("Pre-processing")

    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-o", "--output", help='output folder', action="store", type="string", dest="output")
    parser.add_option("-i", "--input", help='ACL Anthology path', action="store", type="string", dest="path")
    parser.add_option("-b", "--begin", help='index first article', action="store", type="int", dest="begin", default=0)
    parser.add_option("-e", "--end", help='index last article', action="store", type="int", dest="end", default=22000)
    parser.add_option("-x", "--xml", help='write xml prefix', action="store_true", dest="xml_prefix", default=False)
    parser.add_option("-m", "--matches", help='use matches', action="store_true", dest="do_matches", default=False)
    parser.add_option("-v", "--verbose", help='verbose', action="store_true", dest="verbose", default=False)

    (options, args) = parser.parse_args()

    if options.output is None or options.path is None:
        print("input/output path missing")
        sys.exit(2)

    main()
