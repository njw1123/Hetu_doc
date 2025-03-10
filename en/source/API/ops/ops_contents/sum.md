#### hetu.sum

```
hetu.sum(input: hetu.tensor, axes: List[int] = [], keepdims: bool = False) -> hetu.tensor
```

Computes the sum of elements along the specified axes of the input tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.
* axes (List[int], optional): A list of axes along which to sum the elements. If empty, the sum is computed globally.
* keepdims (bool, optional): Whether to keep the dimensions of the input tensor. Default is False.

**Returns:**

* hetu.tensor: A tensor containing the sum of elements along the specified axes.


