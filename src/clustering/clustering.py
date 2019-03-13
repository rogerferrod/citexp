from scipy import sparse
from optparse import OptionParser
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import xml.etree.ElementTree as ET
import matplotlib.cm as cm
import numpy as np

import logging
import codecs
import random
import pickle
import time
import sys


def clustering(matrix):
    """ Clustering
    Applies SVD reduction and clustering algorithm (KMeans) to the tf-idf matrix

    """

    global options

    # svd reduction
    logging.info('svd reduction...')
    svd = TruncatedSVD(options.components, algorithm='arpack')
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    t0 = time.time()
    matrix = lsa.fit_transform(matrix)  # features = terms, sample = sentence
    logging.info("done in %0.3fs" % (time.time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    logging.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    logging.info('matrix shape ' + str(matrix.shape))

    # kmeans
    logging.info('kmeans...')
    km = MiniBatchKMeans(n_clusters=options.n_clusters, init='k-means++', max_iter=100, n_init=6, batch_size=1000)

    t0 = time.time()
    cluster_labels = km.fit_predict(matrix)  # return index of the cluster each sample belongs to
    logging.info("done in %0.3fs" % (time.time() - t0))

    centroids = svd.inverse_transform(km.cluster_centers_)  # original space

    return cluster_labels, matrix, centroids


def silhouette(matrix, cluster_labels, exclude):
    """ Computes silhouette values.
     computes the silhouette coefficient excluding clusters in 'exclude' list

     """

    logging.info('exclude ' + str(exclude) + ' clusters from silhouette')

    # list of (cluster, index_sample)
    data = [(cluster_labels[i], i) for i in range(len(cluster_labels)) if cluster_labels[i] not in exclude]
    # samples foreach cluster
    data_matrix = [[y[1] for y in data if y[0] == x] for x in range(options.n_clusters)]

    # sample_matrix (n_cluster X max{sample_size, row_size})
    sample_matrix = []
    for c in data_matrix:
        sample_matrix.append(random.sample(c, options.sample_size) if len(c) > options.sample_size else c)

    # label for each sample
    samples = []
    for c in sample_matrix:
        samples.extend(c)

    samples.sort()
    labels = [cluster_labels[x] for x in samples]  # index of the cluster each sample belongs to

    deleted_index = [x for x in range(len(cluster_labels)) if x not in samples]  # deleted index_sample
    filtered_matrix = np.delete(matrix, deleted_index, axis=0)  # filtered_matrix (n_samples X n_features)
    logging.info('sampled matrix shape ' + str(filtered_matrix.shape))

    sil_avg = silhouette_score(filtered_matrix, labels)
    logging.info('silhouette average ' + str(sil_avg))
    sample_silhouette_values = silhouette_samples(filtered_matrix, labels)  # silhouette scores for each sample

    ith_silhouettes = []
    # Aggregate the silhouette scores for samples in each cluster
    for i in range(options.n_clusters):
        ith_cluster_silhouette_values = []
        for j in range(len(labels)):
            if labels[j] == i:
                ith_cluster_silhouette_values.append(sample_silhouette_values[j])
        ith_cluster_silhouette_values = np.asarray(ith_cluster_silhouette_values)
        ith_cluster_silhouette_values.sort()

        ith_silhouettes.append(ith_cluster_silhouette_values)

    return filtered_matrix, labels, sil_avg, ith_silhouettes


def cluster_plot(matrix, cluster_labels):
    """ draws 3D plot """

    logging.info('Plot...')
    t0 = time.time()

    svd = TruncatedSVD(n_components=3)
    pos = svd.fit_transform(matrix)

    fig = plt.figure(0)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    colors = cm.rainbow(np.linspace(0, 1, options.n_clusters))
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=cluster_labels))
    groups = df.groupby('label')

    ax = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        ax.plot(group.x, group.y, group.z, marker='o', linestyle='', ms=5, color=colors[name],
                mec='none')
        ax.set_aspect('auto')

    ax.set_xlabel(r'Coordinate 1', fontsize=15)
    ax.set_ylabel(r'Coordinate 2', fontsize=15)
    ax.set_zlabel(r'Coordinate 3', fontsize=15)
    ax.set_title('Tf-idf Matrix (SVD)')
    ax.grid(True)

    # draw and save
    fig.tight_layout()
    plt.savefig(options.output + '/plot.png', bbox_inches='tight')

    logging.info("done in %0.3fs" % (time.time() - t0))


def silhouette_plot(matrix, cluster_labels, exclude):
    """ Computes and draws silhouette plot.
    computes the silhouette coefficient excluding clusters in 'exclude' list

    """

    logging.info('Silhouette...')
    t0 = time.time()

    try:
        # silhouette values
        (sample_matrix, labels, sil_avg, ith_silhouettes) = silhouette(matrix, cluster_labels, exclude)
    except Exception as e:
        logging.error('silhouette values ' + str(e))
        return

    # plot
    fig = plt.figure(1)
    fig.set_size_inches(10, options.n_clusters)
    ax = fig.add_subplot(111)

    ax.set_xlim([-0.3, 0.6])
    ax.set_ylim([0, len(sample_matrix) + (options.n_clusters + 1) * 10])
    y_lower = 10

    for i in range(options.n_clusters):
        ith_cluster_silhouette_values = ith_silhouettes[i]
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.Spectral(float(i) / options.n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                         edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=sil_avg, color="red", linestyle="--")
    ax.set_yticks([])

    # draw and save
    plt.savefig(options.output + '/plot2.png')

    logging.info("done in %0.3fs" % (time.time() - t0))


def remove_all_zero(matrix, snippets):
    """ removes all zero rows from matrix """

    logging.info("remove all_zero")
    t0 = time.time()

    non_zero = np.unique(matrix.nonzero()[0])
    matrix = matrix[non_zero]
    snippets = np.array(snippets)
    snippets = snippets[non_zero]
    snippets = list(snippets)

    logging.info("done in %0.3fs" % (time.time() - t0))

    return matrix, snippets


def main():
    global options

    verbose_output_limit = 10
    output_log = options.output + "/clustering.log"
    output_file = options.output + "/clusters.xml"
    verbose_file = options.output + "/clusters.txt"
    logging.basicConfig(filename=output_log, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    out = open(output_file, "wb")
    out_verbose = codecs.open(verbose_file, "w", "utf-8")

    # load data
    logging.info("loading...")
    matrix = sparse.load_npz(options.input + '/matrix.npz')
    with open(options.input + '/snippet_dict.pkl', 'rb') as f:
        snippet_dict = pickle.load(f)
    with open(options.input + '/features.pkl', 'rb') as f:
        features = pickle.load(f)

    # remove all zero row from matrix
    logging.info(matrix.shape)
    snippets_items = list(snippet_dict.items())
    matrix, snippets_items = remove_all_zero(matrix, snippets_items)
    snippet_dict = dict(snippets_items)

    sentences = list(snippet_dict.values())
    snippets = list(snippet_dict.keys())

    logging.info('num_sentences ' + str(len(sentences)))
    logging.info('matrix shape ' + str(matrix.shape))

    # clustering
    (cluster_labels, matrix, centroids) = clustering(matrix)

    # verbose output
    citations = {'snippet': snippets, 'cluster': cluster_labels}
    frame = pd.DataFrame(citations, index=[cluster_labels], columns=['snippet', 'cluster'])

    out_verbose.write('Num snippets per cluster: \n')
    out_verbose.write(str(frame['cluster'].value_counts()) + '\n')

    # output
    try:
        logging.info('write...')
        t0 = time.time()
        out.write(b'<clusters>')
        for i in range(options.n_clusters):
            xml_cluster = ET.Element('cluster')
            xml_cluster.set('label', str(i))
            list_snippets = frame.ix[i]['snippet'].values.tolist()
            xml_cluster.set('size', str(len(list_snippets)))
            for snippet in list_snippets:
                xml_snippet = ET.SubElement(xml_cluster, 'snippet')
                xml_snippet.text = snippet

            tree = ET.ElementTree(xml_cluster)
            tree.write(out, encoding='utf-8')
        out.write(b'</clusters>')
        logging.info("done in %0.3fs" % (time.time() - t0))
    except Exception as e:
        logging.error('output ' + str(e))

    # verbose output
    if options.use_verbose:
        try:
            out_verbose.write('Top terms per cluster: \n\n')
            order_centroids = centroids.argsort()[:, ::-1]
            for i in range(options.n_clusters):
                out_verbose.write("Cluster " + str(i) + '\n')
                for ind in order_centroids[i, :verbose_output_limit]:
                    out_verbose.write(' ' + str(features[ind]))
                out_verbose.write('\n')
                out_verbose.write("Cluster " + str(i) + " articles: \n")
                for snippet in frame.ix[i]['snippet'].values.tolist()[:verbose_output_limit]:
                    out_verbose.write(' ' + str(snippet))
                out_verbose.write('\n')
                out_verbose.write("Cluster " + str(i) + " sentences: \n")
                for snippet in frame.ix[i]['snippet'].values.tolist()[:verbose_output_limit]:
                    out_verbose.write("  -" + snippet_dict[snippet] + '\n')
                out_verbose.write('\n')
            out_verbose.write('\n')
        except Exception as e:
            logging.error('verbose ' + str(e))

    # plot
    try:
        cluster_plot(matrix, cluster_labels)
    except Exception as e:
        logging.error('cluster_plot ' + str(e))

    # silhouette plot
    try:
        lista = []
        for i in range(options.n_clusters):
            lista.append(len(frame.ix[i]['snippet'].values.tolist()))
        ex = max(lista)
        exclude = lista.index(ex)
        silhouette_plot(matrix, cluster_labels, [exclude])
    except Exception as e:
        logging.error('silhouette_plot ' + str(e))


if __name__ == "__main__":
    print("Clustering")

    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-i", "--input", help='input folder', action="store", type="string", dest="input")
    parser.add_option("-o", "--output", help='output folder', action="store", type="string", dest="output")
    parser.add_option("-c", "--clusters", help='num clusters', action="store", type="int", dest="n_clusters")
    parser.add_option("-r", "--reduction", help='svd components', action="store", type="int", dest="components")
    parser.add_option("-v", "--verbose", help='verbose output', action="store_true", dest="use_verbose", default=False)
    parser.add_option("-s", "--silhouette", help='silhouette sample', action="store", type="int", dest="sample_size")

    (options, args) = parser.parse_args()

    if options.input is None or options.output is None or options.n_clusters is None \
            or options.n_clusters is None or options.components is None or options.sample_size is None:
        print("Missing parameters")
        sys.exit(2)

    main()
