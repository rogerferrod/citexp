# Results

The results of the computation, for each step, are reported in the <i>output.rar</i> file,
meanwhile the graph is represented by the four csv files that can be directly used
to import the graph into a database.

## Graph
| File name                 | Description   |
| :------------------------ | :-------------|
| articles.csv 	       | contains the "Article" nodes, identified by an alphanumeric id, with all the information available about it
| objects.csv           | contains the "Semantic Note" nodes associated with the article. <br> Semantic notes contain information about the reasons why the article was cited and the context of the citation
| citations.csv     | contains the edges representing the citations witch link two articles, it also contains all the available information about the snippet, including the type of citation
| purposes.csv      | contains the edges, witch link the semantic notes with the corresponding articles, labeled with the type of citation

## Output
| File name                 | Description   |
| :------------------------ | :-------------|
| class.json 	       | it is the dictionary that associates the labels with the corresponding clusters
| clusters.txt      | contains some analytics about the clustering, such as the top terms for each cluster and the cardinality of clusters
| clusters.xml      | contains all the snippets collected in clusters
| plot.png          | a portion of the dataset reduced to 3 dimensions and divided into clusters
| plot2.png         | the silhouette analysis plot
| preprocessing.xml | the result of the preprocessing step containing all the articles, snippets and references of the original dataset