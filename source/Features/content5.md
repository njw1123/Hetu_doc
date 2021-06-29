High-performance GNN Training
=============================


Graph Neural Networks (GNNs) are powerful and flexible neural networks that use the naturally sparse connectivity information of the data. However, the sparsity and irregularity in graphs make it notoriously difficult to perform efficient GNN computing on data parallel hardware like GPU. Our system uses PCGCN, a partition-centric GCN framework that enhances cache-efficiency and memory performance to efficiently execute GCNs. PCGCN executes a GCN model as bulk synchronous steps of message exchange between vertex subsets called partitions. PCGCN propagates updates from nodes to partitions and reduces the redundancy associated with vertex-centric processing model by leveraging the graph locality and the cache We also adopt a hybrid partition-centric processing strategy that can adaptively select the optimal mode for each pair of graph partitions with different computation characteristics. In these ways, PCGCN extracts high performance from the memory hierarchy. In this system, we use Metis to partition the graph.

We compare Hetu to Deep Graph Library(DGL)  and PyTorch Geometric (PyG), meanwhile, we selected 9 datasets, including 5 smaller datasets: PubMed, Cora, citeser, Coauthor_ phy, Blogcatalog, four large datasets: Reddit, Proteins, ArXiv, Amazon 0601. The GCN network with 64 hidden layer nodes and 4 hidden layers is used for training. The results are as follows.

|                | Pubmed  | Cora    | Citeseer | Coauthor_phy | Blogcatalog |
|----------------|---------|---------|----------|--------------|-------------|
| DGL            | 0.0118s | 0.0118s | 0.0118s  | 0.0228s      | 0.0128s     |
| PYG            | 0.0053s | 0.0057s | 0.0054s  | 0.0195s      | 0.0078s     |
| Hetu           | 0.0016s | 0.0010s | 0.0011s  | 0.0059s      | 0.0022s     |
| Hetu(reorder)  | 0.0016s | 0.0009s | 0.0011s  | 0.0056s      | 0.0023s     |
| Hetu(hybrid)   | 0.0024s | 0.0011s | 0.0012s  | 0.0062s      | 0.0022s     |


It can be seen from the table that for the small graph, the calculation speed of Hetu is faster than the other two graph neural network frameworks, and only using the optimized spmm can get better training speed, but there is no advantage in the hybrid mode.

|                | Reddit  | Proteins | Arxiv   | Amazon0601 |
|----------------|---------|----------|---------|------------|
| DGL            | 0.6135s | 0.2431s  | 0.0392s | 0.0751s    |
| PYG            | oom     | oom      | 0.0605s | 0.1264s    |
| Hetu           | 0.3442s | 0.1850s  | 0.0195s | 0.0302s    |
| Hetu(reorder)	 | 0.1389s | 0.0843s  | 0.0174s | 0.0196s    |
| Hetu(hybrid)	  | 0.1000s | 0.0575s  | 0.0208s | 0.0262s    |

For dense large graph, the hybrid mode can speed up the training. If the nodes in the graph are sparse, the optimal training speed can be obtained by partition the graph.

Here is a simple Python demo to show how to train GNN model.

### Load dataset.

First, you need to load dataset from your file folder.If you want, you can reorder the graph to accelerate computing.
```python
import numpy as np
import scipy.sparse as sp
from GNN.graph import metis_reorder

dir_name = 'your_dir'
adj = sp.load_npz(dir_name+"adj.npz").tocoo()
features = np.load(dir_name+"features.npy")
labels = np.load(dir_name+"labels.npy")
adj, features, labels = metis_reorder(adj, features, labels)
```

### Set models.

Then, you can get the parameters of graph and define variables  you need. After that, you can make your own gcn model.

```python
from hetu import initializers
from hetu import optimizer
from hetu import gpu_ops as ad

node_count = adj.shape[0]
num_features = features.shape[1]
num_classes = np.max(labels)+1
hidden_size = 64
 
ctx = ndarray.gpu(0)
A = ad.Variable(name="A", trainable=False)
H = ad.Variable(name="H")
W1 = initializers.xavier_uniform(shape=(num_features, hidden_size), name="W1", trainable=True, ctx=ctx)
W2 = initializers.xavier_uniform(shape=(hidden_size, num_classes), name="W2", trainable=True, ctx=ctx)
y_ = ad.Variable(name="y_")       
 
z1 = ad.matmul_op(H,W1,ctx=ctx)  
z2 = ad.spmm_op(A, z1,ctx=ctx) 
z3 = ad.relu_op(z2,ctx=ctx)       
z4 = ad.matmul_op(z3,W2,ctx=ctx)       
y = ad.spmm_op(A, z4,ctx=ctx)  
loss = ad.softmaxcrossentropy_op(y, y_,ctx=ctx)   
opt = optimizer.AdamOptimizer()
train_op = opt.minimize(loss)
```

### Train models.

Finally, you can set the executor and train your models.

```python
from hetu import ndarray

executor = ad.Executor([y,loss,train_op], ctx=ctx)
def convert_to_one_hot(vals, max_val = 0):
    #Helper method to convert label array to one-hot array
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals   

feed_dict = {
  H: ndarray.array(features, ctx=ctx),
  y_ : ndarray.array(convert_to_one_hot(labels, max_val=num_classes), ctx=ctx),
  A: ndarray.sparse_array(adj.data,(adj.row,adj.col),shape=(adj.shape),ctx=ctx)
}
    
epoch_num = 100
for i in range(epoch_num):
    results = executor.run(feed_dict = feed_dict)  
    y_predict = results[0].asnumpy().argmax(axis=1)
    loss = results[1].asnumpy().mean()
    acc = float(np.sum(y_predict==labels)/labels.shape[0])
    print("Epoch :%d , loss:%.4f , acc:%.4f "%(i,loss,acc))
```

If you want to use the hybrid mode, here is the simple Python demo.

### Load dataset.

To use the hybird mode, you need to split the graph to get the block-sparse part and coo part of the adjacency matrix. Also, you can set the block size and density threshold.
```python
import numpy as np
import scipy.sparse as sp
from GNN.graph import metis_reorder
from GNN.graph import split_graph

dir_name = 'your_dir'
adj = sp.load_npz(dir_name+"adj.npz").tocoo()
features = np.load(dir_name+"features.npy")
labels = np.load(dir_name+"labels.npy")
adj, features, labels = metis_reorder(adj, features, labels)
block_size = 32
theta = 0.05
layout, dense_matrix, sparse_coo, features = split_graph(adj, features, block_size = block_size, theta = theta)
```

### Set models.

The two parts of the sparse matrix are multiplied separately. After multiplication, add them to get the full result.

```python
from hetu import initializers
from hetu import optimizer
from hetu import gpu_ops as ad

node_count = labels.shape[0]
num_features = features.shape[1]   
num_classes = np.max(labels)+1
hidden_size = 64
   
ctx = ndarray.gpu(0)
A = ad.Variable(name="A",trainable=False)
H = ad.Variable(name="H")
W = ad.Variable(name="W")
W1 = initializers.xavier_uniform(shape=(num_features, hidden_size), name="W1", trainable=True, ctx=ctx)
W2 = initializers.xavier_uniform(shape=(hidden_size, num_classes), name="W2", trainable=True, ctx=ctx)    
y_ = ad.Variable(name="y_")    
   
z1 = ad.matmul_op(H,W1,ctx=ctx)  
z2_dense = ad.bsmm_op(z1,W,layout,block_size,True,ctx)
z2_sparse = ad.spmm_op(A, z1,ctx=ctx) 
z2 = ad.add_op(z2_dense, z2_sparse)
z3 = ad.relu_op(z2,ctx=ctx)       
z4_dense = ad.bsmm_op(z3,W,layout,block_size,True,ctx)   
z4_sparse = ad.spmm_op(A, z3,ctx=ctx)
z4 = ad.add_op(z4_dense, z4_sparse)        
y = ad.matmul_op(z4,W2,ctx=ctx)    
yy = ad.slice_op(y, (0,0), (node_count,num_classes), ctx)
loss = ad.softmaxcrossentropy_op(yy, y_,ctx=ctx)   
opt = optimizer.AdamOptimizer()
train_op = opt.minimize(loss)
```

### Train models.

You need to send both the sparse part of the adjacency matrix and the block-sparse part into the feed dictionary.

```python
from hetu import ndarray

executor = ad.Executor([yy,loss,train_op], ctx=ctx)
feed_dict = {
    H: ndarray.array(features, ctx=ctx),
    W : ndarray.array(dense_matrix, ctx=ctx),
    y_ : ndarray.array(convert_to_one_hot(labels, max_val=num_classes), ctx=ctx),
    A: ndarray.sparse_array(sparse_coo.data,(sparse_coo.row,sparse_coo.col),shape=(sparse_coo.shape),ctx=ctx),
}

epoch_num = 100
for i in range(epoch_num):
    results = executor.run(feed_dict = feed_dict)  
    y_predict = results[0].asnumpy().argmax(axis=1)
    loss = results[1].asnumpy().mean()
    acc = float(np.sum(y_predict==labels)/labels.shape[0])
    print("Epoch :%d , loss:%.4f , acc:%.4f "%(i,loss,acc))
```
