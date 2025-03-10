### nn
#### HtMultiParallelRMSNorm

```python
class HtMultiParallelRMSNorm(normalized_shape, multi_ds_parallel_config, sequence_parallel=False, recompute_allgather=False, dtype=hetu.float32, name='rmsnorm')
```
Define distributed RMSNorm based on parameters.

**Parameters:**
- normalized_shape (Union[int, List[int]]): Specifies the dimensions to be normalized.
- multi_ds_parallel_config (List[Dict]): Configuration of the RMSNorm extracted from the DistConfig file.
- sequence_parallel (bool): Determines whether to enable Megatron sequence parallelism.
- recompute_allgather (bool): When both recomputation and sequence_parallel are enabled, this parameter manually configures the recomputation of communication.
- dtype (hetu.dtype): Specifies the precision type of parameters.
- name (str): Name of the module.
	
**Methods:**
- forward(input_p): Performs the RMSNorm computation.

#### HtMultiParallelLayerNorm

```python
class HtMultiParallelLayerNorm(normalized_shape, multi_ds_parallel_config, sequence_parallel=False, eps=1e-5, dtype=hetu.float32, name='ln')
```
Define Distributed LayerNorm Based on Parameters

**Parameters:**
- normalized_shape (Union[int, List[int]]): Specifies the dimensions to be normalized.
- multi_ds_parallel_config (List[Dict]): Configuration of the LayerNorm extracted from the DistConfig file.
- sequence_parallel (bool, optional): Determines whether to enable Megatron sequence parallelism. Defaults to False.
- eps (float, optional): A small constant added to the standard deviation to prevent division by zero and ensure numerical stability. Defaults to 1e-5.
- dtype (hetu.dtype, optional): Specifies the precision type of parameters. Defaults to hetu.float32.
- name (str, optional): Name of the module. Defaults to 'ln'.

**Methods:**
- forward(input_p): Performs the LayerNorm computation.


#### HtMultiParallelEmbedding

```python
class HtMultiParallelEmbedding(num_embeddings, embedding_dim, multi_ds_parallel_config, init_method='xavier_normal_', dtype=hetu.float32, name='embedding')
```

The embedding will be redundantly and fully stored across all devices.e.g., position embedding.

**Parameters:**
- num_embeddings (int): The length of the embedding table. For position embedding, this corresponds to the maximum input length.
- embedding_dim (int): The hidden dimension of the embedding table.
- multi_ds_parallel_config (List[Dict]): Configuration of the embedding extracted from the DistConfig file.
- init_method (str): Initialization method.
- dtype (hetu.dtype): Specifies the precision type of parameters.
- name (str): Name of the module.


**Methods:**
- forward(input_p): Performs the computation, extracting the corresponding row from the embedding table.



#### HtMultiVocabParallelEmbedding

```
class HtMultiVocabParallelEmbedding(num_embeddings, embedding_dim, multi_ds_parallel_config, init_method='xavier_normal_', dtype=hetu.float32, name='vocab_embedding')
```

The embedding table is partitioned row-wise within the tensor parallel (TP) group.


**Parameters:**
- num_embeddings (int): The length of the embedding table. 
- embedding_dim (int): The hidden dimension of the embedding table.
- multi_ds_parallel_config (List[Dict]): Configuration of the embedding extracted from the DistConfig file.
- init_method (str): Initialization method.
- dtype (hetu.dtype): Specifies the precision type of parameters.
- name (str): Name of the module.
	
**Methods:**
- forward(input_p): Performs the embedding lookup by extracting the rows corresponding to input_p from the embedding table and aggregating them within the tensor parallel (TP) group.


#### HtMultiColumnParallelLinear

```
class HtMultiColumnParallelLinear(in_features, out_features, multi_ds_parallel_config,bias=True, gather_output=True, init_method='xavier_normal_', dtype=hetu.float32, name='colp')
```

Matrix multiplication with column-parallel (CP) in tensor parallelism.

**Parameters:**
- in_features (int): The input dimension.
- out_features (int): The output dimension.
- multi_ds_parallel_config (List[Dict]): Configuration of this module extracted from the DistConfig file.
- bias (bool): Whether to include a bias term.
- gather_output (bool): Whether to aggregate the output.
- init_method (str): The initialization method.
- dtype (hetu.dtype): Specifies the precision type of parameters.
- name (str): The name of the module.

**Methods:**
- forward(input_p): Performs the matrix multiplication calculation.



#### HtMultiRowParallelLinear

```
class HtMultiRowParallelLinear(in_features, out_features, multi_ds_parallel_config, sequence_parallel=False, bias=True, init_method='xavier_normal_', dtype=hetu.float32, name='rowp')
```

Row-Sliced Matrix Multiplication in Tensor Parallelism


**Parameters:**
- in_features (int): The input dimension.
- out_features (int): The output dimension.
- multi_ds_parallel_config (List[Dict]): Configuration details for this module, extracted from the DistConfig file.
- bias (bool): Whether to include a bias term.
- sequence_parallel (bool): Whether to enable sequence parallelism.
- init_method (str): The initialization method for the parameters.
- dtype (hetu.dtype): The data type for parameter precision.
- name (str): The module name.

**Methods:**
- forward(input_p): Performs the matrix multiplication operation.

#### Dropout

```
class Dropout(p: float = 0.5, inplace: bool = False)
```

Defines a Dropout module based on parameters.

**Parameters:**
- p (float): The probability of neurons being dropped in the dropout layer.
- inplace (bool): Whether the output shares the same memory space as the input.

**Methods:**
- forward(input): Performs the dropout operation.

#### NewGeLU，ReLU，Tanh，Sigmoid

```
class activation_func(multi_ds_parallel_config = None, inplace: bool = False)
```

Includes four activation functions: NewGeLU, ReLU, Tanh, and Sigmoid, all sharing the same parameter structure.

**Parameters:**
- multi_ds_parallel_config (List[Dict]): Configuration details for this module, extracted from the DistConfig file.
- inplace (bool): Whether the output shares the same memory space as the input.

**Methods:**
- forward(input): Executes the corresponding activation function computation.


#### ModuleList

```
class ModuleList(modules: Optional[Iterable[Module]] = None)
```

Ordered Container for Storing Multiple Modules in a Neural Network


**Parameters:**
- modules (Optional[Iterable[Module]]): An iterable (e.g., list, tuple, set, etc.), where each element is an instance of the Module class.

