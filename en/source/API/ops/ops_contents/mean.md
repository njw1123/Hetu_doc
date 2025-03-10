#### hetu.mean

```
hetu.mean(input: hetu.tensor, axes: List[int] = [], keepdims: bool = False) -> hetu.tensor
```

Computes the mean of elements along the specified axes of the input tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.
* axes (List[int], optional): A list of axes along which to compute the mean. If empty, the mean is computed globally.
* keepdims (bool, optional): Whether to keep the dimensions of the input tensor. Default is False.

**Returns:**

* hetu.tensor: A tensor containing the mean of elements along the specified axes.


