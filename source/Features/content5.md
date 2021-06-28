High-performance GNN Training
=============================


Graph Neural Networks (GNNs) are powerful and flexible neural networks that use the naturally sparse connectivity information of the data. However, the sparsity and irregularity in graphs make it notoriously difficult to perform efficient GNN computing on data parallel hardware like GPU. Our system uses PCGCN, a partition-centric GCN framework that enhances cache-efficiency and memory performance to efficiently execute GCNs. PCGCN executes a GCN model as bulk synchronous steps of message exchange between vertex subsets called partitions. PCGCN propagates updates from nodes to partitions and reduces the redundancy associated with vertex-centric processing model by leveraging the graph locality and the cache We also adopt a hybrid partition-centric processing strategy that can adaptively select the optimal mode for each pair of graph partitions with different computation characteristics. In these ways, PCGCN extracts high performance from the memory hierarchy. In this system, we use Metis to partition the graph.

We compare Hetu to Deep Graph Library(DGL)  and PyTorch Geometric (PyG) and selected 9 datasets, including 5 smaller datasets: PubMed, Cora, citeser, coauthor_ phy, Blogcatalog, four large datasets: Reddit, Proteins, ArXiv, Amazon 0601. The GCN network with 64 hidden layer nodes and 4 hidden layers is used for training. We use a four layers GCN as the model, and the number of hidden layer nodes is 64. The results are as follows.

|                | Pubmed  | Cora    | Citeseer | Coauthor_phy | Blogcatalog |
|----------------|---------|---------|----------|--------------|-------------|
| DGL            | 0.0118s | 0.0118s | 0.0118s  | 0.0228s      | 0.0128s     |
| PYG            | 0.0053s | 0.0057s | 0.0054s  | 0.0195s      | 0.0078s     |
| Hetu           | 0.0016s | 0.0010s | 0.0011s  | 0.0059s      | 0.0022s     |
| Hetu(reorder)  | 0.0016s | 0.0009s | 0.0011s  | 0.0056s      | 0.0023s     |
| Hetu(hybrid)   | 0.0024s | 0.0011s | 0.0012s  | 0.0062s      | 0.0022s     |


It can be seen from the table that for the small graph, the calculation speed of Hetu is faster than the other two graph neural network frameworks, and only using the optimized spmm can get better training speed, but there is no advantage in the hybrid mode.

+------------------------------------------------------------------------------+
|                  | Reddit       | Proteins     | Arxiv        | Amazon0601   | 
+------------------+--------------+--------------+--------------+--------------+
| DGL              | 0.6135s      | 0.2431s      | 0.0392s      | 0.0751s      | 
+------------------+--------------+--------------+--------------+--------------+
| PYG              | OOM          | OOM          | 0.0605s      | 0.1264s      | 
+------------------+--------------+--------------+--------------+--------------+
| Hetu             | 0.3442s      | 0.1850s      | 0.0195s      | 0.0302s      |
+------------------+--------------+--------------+--------------+--------------+
| Hetu(reorder)    | 0.1389s      | 0.0843s      | 0.0174s      | 0.0196s      |
+------------------+--------------+--------------+--------------+--------------+
| Hetu(hybrid)     | 0.1000s      | 0.0575s      | 0.0208s      | 0.0262s      |
+------------------+--------------+--------------+--------------+--------------+

For dense large graph, the hybrid model can speed up the training. If the nodes in the graph are sparse, the optimal training speed can be obtained by partition the graph.




