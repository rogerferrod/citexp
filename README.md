# CitExp

<b>Informed Exploration of Scientific Literature via Semantically-Enriched Citation Paths </b><br>
Citexp provides a set of natural language analysis tools, written in python,
for the construction of a semantically-enriched citation graph that makes use of
Natural Language Processing and Data Mining technologies to enable advanced
retrieval and exploration of a scientific literature. This release is specific for the ACL Antholgy, a corpus of scientific
publications sponsored by the Association for Computational Linguistics. <br/>
A simple web application, that allows the navigation of the resulting graph,
is available at http://citexp.di.unito.it

## Getting Started

The project is subdivided in four sequential steps, these instructions will get
you a copy of the project up and running on your local machine for development
and testing purposes. <br />
Software has been tested using the ACL anthology, consequently the <i>preprocessing</i> phase is specific for this corpus.
You can download a copy of the corpus here: https://acl-arc.comp.nus.edu.sg/

### Prerequisites

The <i>preprocessing</i> phase requires metadata of the corpus and the output of <i>Parscit</i> tool, both of them are included in the dataset. <br/>
The <i>vectorizer</i> phase requires that the StanfordCoreNLP server is running, the tool can been found here: https://stanfordnlp.github.io/CoreNLP/index.html <br>
The server, for example, can be started as follow:
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### Requirements

* [Python](https://www.python.org/) - Python 3.6.6
* [Java](https://www.java.com/it/download/) - Java 8
* [nltk](https://www.nltk.org/) - NLTK
* [numpy](http://www.numpy.org/) - NumPy
* [scipy](https://www.scipy.org/) - SciPy
* [scikit-learn](https://scikit-learn.org/stable/) - Scikit-learn
* [matplotlib](https://matplotlib.org/) - Matplotlib
* [lxml](https://lxml.de/) - lxml
* [pandas](https://pandas.pydata.org/) - Pandas
* [requests](http://docs.python-requests.org/en/master/) - Requests 2.21.0
* [pathlib](https://docs.python.org/3/library/pathlib.html) - Pathlib
* [progressbar2](https://pypi.org/project/progressbar2/) - Progressbar 2 3.39.2

## Pipeline

The pipeline consist of: <i>preprocessing, vectorizer, clustering</i> and <i>graph construction. </i>

### Preprocessing

The required arguments for the <i>preprocessing</i> script are:
```
-o --output        Directory used to store the output
-i --input        ACL Anthology directory path
```

Optional arguments:

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| -b --begin 	       |	0           |index of first article to consider
| -e  --end          | 22000           |index of last article to consider
| -x -–xml 	       |	False	            |write xml prefix in the output file
| -m --matches  		       | False	           | find matches beetween citations and articles in metadata
| -v --verbose 		           | False             | print detailed log

Example:
```
python preprocessing.py -o ../../output -i ../../resources/ACL_Anthology -b 0 -e 1000 -m
```

### Vectorizer

Before starting the <i>vectorizer</i> make sure the StanfordCoreNLP server is running.
The required arguments for the <i>vectorizer</i> script are:
```
-o --output        Directory used to store the output
-i --input       Preprocessing's file to be used as input
-u --url        StanfordCoreNLP server url
```

Optional arguments:

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| -d --depth 	       |	2           |maximum depth reachable during the visit of the dependency graph
| -l  --limit          | -1           |maximum number of input snippets to be considered

Example:
```
python vectorizer.py -i "../../output/preprocessing.xml" -o "../../output" -d 2 -u "http://localhost:9000"
```

### Clustering

The required arguments for the <i>clustering</i> script are:
```
-o --output        Directory used to store the output
-i --input       Directory containing the output of the vectorizer step
-c --clusters        Number of clusters
-r --reduction        Number of SVD compontents
-s --silhouette        Size of samples for the silhouette computation
```

Optional arguments:

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| -v --verbose 		           | False             | print detailed log

Example:
```
python clustering.py -i "../../output" -o "../../output" -c 30 -r 3000 -v -s 1000
```

### Graph construction
In order to labeling the clusters is required a json file with the label for each cluster to be considered.
Example of the json file:
```
{
	"1": "see for details",
	"2": "present",
	"5": "use",
	"10": "report",
	"11": "proposed by",
	"12": "method",
	"17": "present",
	"20": "follow",
	"24": "approach",
	"25": "introduce"
}
```

The required arguments for the <i>graph</i> script are:
```
-o --output        Directory used to store the output
-c --clusters       Clustering's file produced by the clustering step (named 'clusters.xml')
-a --aclpath        ACL Anthology directory path
-p --preprocessing        Preprocessing's file produced by the proprocessing step (named 'proprocessing.xml')
-j --json        Json file containing the label for each clusters to be considered
```

Example:
```
python graph.py -c "../../output/clusters.xml" -a ../../resources/ACL_Anthology -p "../../output/preprocessing.xml" -j "../../output/class.json" -o "../../output"
```


## Results
Detailed information about the results are provided by [README.md](results/README.md)

## Publications
Roger Ferrod, Claudio Schifanella, Luigi Di Caro, Mario Cataldi.: Disclosing Citation Meanings for Augmented Research Retrieval and Exploration.: In Proceedings of 16th International Conference, ESWC 2019, Portorož, Slovenia, 2nd - 6th June, 2019
</br>
https://link.springer.com/chapter/10.1007/978-3-030-21348-0_7

## Authors

* **Roger Ferrod** - *Initial work* - [roger.ferrod@edu.unito.it](mailto:roger.ferrod@edu.unito.it)

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Supervisors:
* Claudio Schifanella
* Luigi Di Caro
