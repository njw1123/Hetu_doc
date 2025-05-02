# hetu.split

```python
hetu.split(input: hetu.tensor, num_chunks: int, dim: int, remain: bool=False) -> List[hetu.tensor]
```

Splits the input tensor evenly into multiple sub-tensors along the specified dimension.

The number of sub-tensors is determined by the `num_chunks` parameter. If the input tensor cannot be evenly divided, the `remain` parameter determines whether to include the remaining elements in the last sub-tensor.

**Parameters:**

* `input (hetu.tensor)`: The input tensor to be split.
* `num_chunks (int)`: The number of sub-tensors to create.
* `dim (int)`: The dimension along which to split the tensor.
* `remain (bool, optional)`: If True, the last sub-tensor will include any remaining elements. If False, the last sub-tensor will be discarded if it cannot be evenly divided. Default is False.

**Returns:**

* `List[hetu.tensor]`: A list of sub-tensors created by splitting the input tensor.

```python
hetu.split(input: hetu.tensor, chunks: List[int], dim: int) -> List[hetu.tensor]
```

Splits the input tensor into multiple sub-tensors based on the provided chunk sizes along the specified dimension.

**Parameters:**

* `input (hetu.tensor)`: The input tensor to be split.
* `chunks (List[int])`: A list of integers specifying the sizes of each sub-tensor. The sum of this list must equal the size of the input tensor along the specified dimension.
* `dim (int)`: The dimension along which to split the tensor.

**Returns:**

* `List[hetu.tensor]`: A list of sub-tensors created by splitting the input tensor based on the provided chunk sizes.

```python
hetu.split(input: hetu.tensor, task_batch_idxs: List[hetu.Tensor], dim: int) -> List[hetu.tensor]
```

Splits the input tensor into multiple sub-tensors based on the provided multi-task batch indices.

**Parameters:**

* `input (hetu.tensor)`: The input tensor to be split.
* `task_batch_idxs (List[hetu.Tensor])`: A list of tensors containing the indices for each task. The length of this list determines the number of sub-tensors created.
* `dim (int)`: The dimension along which to split the tensor.

**Returns:**

* `List[hetu.tensor]`: A list of sub-tensors created by splitting the input tensor based on the provided indices.
