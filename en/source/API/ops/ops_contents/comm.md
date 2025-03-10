#### hetu.comm

```
hetu.comm(input: hetu.tensor, dst_ds_hierarchy: List[hetu.DistributedStatesUnion], mode: str) -> hetu.tensor
```

Performs a communication operation involving collective reductions, with the communication type determined internally by Hetu based on the ds_hierarchy of the input and the dst_ds_hierarchy.

**Parameters:**

* input (hetu.tensor): The input tensor to be communicated.

* dst_ds_hierarchy (List[hetu.DistributedStatesUnion]):The target distributed states, where each state in the list represents the target distributed state in a specific strategy. This is because Hetu supports switching between different strategies.

* mode (str): The reduction operation type, which must be one of the following: “sum”，“mean”，“prod”，“max”，“min”，“none”.

**Returns:**

* hetu.tensor: The tensor resulting from the reduction operation.

```
hetu.comm(input: hetu.tensor, dst_ds_hierarchy: List[hetu.DistributedStatesUnion], dst_dg_hierarchy: hetu.DeviceGroupHierarchy, is_pipeline_op: bool = True) -> hetu.tensor
```

Performs communication for sending and receiving information across distributed nodes and device groups.

**Parameters:**

* input (hetu.tensor): The input tensor to be communicated.

* dst_ds_hierarchy (List[hetu.DistributedStatesUnion]):The target distributed states, where each state in the list represents the target distributed state in a specific strategy. This is because Hetu supports switching between different strategies.

* dst_dg_hierarchy (hetu.DeviceGroupHierarchy): The destination device group.

* is_pipeline_op (bool, optional): Whether the communication is between different stages of a pipeline. Default is True.

**Returns:**

* hetu.tensor: The tensor receiving the communication result.

```
hetu.comm(input: hetu.tensor, dst_ds_hierarchy: List[hetu.DistributedStatesUnion]) -> hetu.tensor
```

Performs collective communication across distributed nodes or devices, with the default reduction type being “sum.”

**Parameters:**

* input (hetu.tensor): The input tensor to be communicated.

* dst_ds_hierarchy (List[hetu.DistributedStatesUnion]):The target distributed states, where each state in the list represents the target distributed state in a specific strategy. This is because Hetu supports switching between different strategies.

**Returns:**

* hetu.tensor: The tensor that receives the communication result.

