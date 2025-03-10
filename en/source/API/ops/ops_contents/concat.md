#### hetu.concat

```
hetu.concat(inputA: hetu.tensor, inputB: hetu.tensor, axis: int) -> hetu.tensor
```

Concatenates two tensors along the specified axis.

**Parameters:**

* inputA (hetu.tensor): The first input tensor.
* inputB (hetu.tensor): The second input tensor.
* axis (int): The axis along which to concatenate (negative indexing is supported, e.g., -1 refers to the last axis).

**Returns:**

* hetu.tensor: A new tensor formed by concatenating the two tensors along the specified axis.

```
hetu.concat(inputs: List[hetu.tensor], axis: int = 0) -> hetu.tensor
```

Concatenates multiple tensors along the specified axis.

**Parameters:**

* inputs (List[hetu.tensor]): A list of tensors to be concatenated (all tensors must have the same dimensions along axes other than the concatenation axis).
* axis (int): The axis along which to concatenate the tensors (default is 0).

**Returns:**

* hetu.tensor: A new tensor formed by concatenating all tensors along the specified axis.

