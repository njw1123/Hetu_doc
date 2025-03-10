
### Graph
Hetu provides several different types of computation graphs, including eager graph, define-by-run graph, define-and-run graph, executable graph and so on. Among these, the eager graph is a dynamic graph, while the define-and-run graph and executable graph are static graphs. The define-and-run graph is used for defining and optimizing the computation graph, while the executable graph focuses on executing the fully optimized computation graph.

```python 
with hetu.graph(g:str, create_new:bool=False, prefix:str="default", tmp:bool=False) as g:
  # code
```
Define the context of the graph, and the operations defined in the context will be automatically associated with the graph.

**Parameters**

- g(str):  The type of the graph execution paradigm, e.g. "eager", "define-by-run", "define-and-run". The executable graph does not require explicit definition on the Python side, but is instead jointly optimized with the define-and-run mechanism in the framework's internal implementation for enhanced training efficiency.
- create_new(bool): Whether to create a new graph.
- prefix(str): The prefix of the graph name.
- tmp(bool): Whether to create a temporary graph.

**Note**：The following sections describe the properties and methods of the graph object itself, not the context manager operations for graph execution.

**Properties**

- name(str): The name of the graph. 
- id(int): The unique identifier for the graph.
- num_strategy(int): The number of strategies available for use in the graph.
- cur_strategy_id (int): The current strategy ID.
- use_hetero_id (bool): Whether to use hetero ID.
- cur_hetero_id (int): The current hetero ID, which refers to the specific pipeline being used.

**Methods**

```
run(fetch:hetu.Tensor, feed_dict:FeedDict=None)->List[hetu.NDArray]
run(fetch:List[Tensor], feed_dict:FeedDict=None)->List[hetu.NDArray]
run(loss:hetu.Tensor, fetch:hetu.Tensor, feed_dict:FeedDict=None, num_micro_batches:int=1)->List[hetu.NDArray]

# Supports hot switching between different strategies, controlled by cur_strategy_id
run(loss:hetu.Tensor, fetch:List[Tensor], feed_dict:FeedDict=None, num_micro_batches:int=1, cur_strategy_id:int=0, run_level:int=0, save_checkpoint:bool=false, grad_scale:double=1)->List[hetu.NDArray]

# Supports different strategies for computation and optimization, controlled by compute_strategy_id and optimize_strategy_id
run(loss:hetu.Tensor, fetch:List[Tensor], feed_dict:FeedDict=None, num_micro_batches:int=1, compute_strategy_id:int=0, 
optimize_strategy_id:int=0, run_level:int=0, save_checkpoint:bool=false, grad_scale:double=1)->List[hetu.NDArray]
```

- **Parameters**
  - fetch(hetu.Tensor, List[hetu.Tensor]): The tensor(s) to be fetched after the graph runs, such as the loss or other outputs.
  - feed_dict(Dict[hetu.Tensor, List[numpy.ndarray]]): A dictionary that contains the input data required for running the graph. 
  - num_micro_batches(int): The number of micro-batches to divide the data into during the graph run
  - cur_strategy_id：The ID of the currently active strategy. Hetu supports dynamic switching between different strategies during execution.
  - compute_strategy_id：The strategy used for computation. Hetu supports different strategies for computation and optimization.
  - optimize_strategy_id：The strategy used for optimization. Like compute_strategy_id, Hetu supports separate strategies for computation and optimization.
  - run_level(hetu.run_level): Defines the level of execution in Hetu:
    - hetu.run_level("update"): Full training, including both gradient computation and weight update.
    - hetu.run_level("grad"): Computes gradients but does not perform the update.
    - hetu.run_level("compute_only"):  Only computes without gradient reduction or updates.
    - hetu.run_level("alloc"): Only allocates memory for the graph.
    - hetu.run_level("topo"): Only builds the execution graph.
  - save_checkpoint：Whether to save the model checkpoint during execution.
  - grad_scale：The factor by which gradients are scaled during training

```
set_num_strategy(num_strategy:int)->int
```

This function sets the number of strategies to be used in the computation graph.

- **Parameters**

  - num_strategy (int): The number of strategies to be set for the graph. 

```
merge_strategy(graph_id:int)->None
```

This function merges strategies associated with a specific computation graph.

- **Parameters**

  - graph_id (int): The ID of the computation graph whose strategies will be merged.