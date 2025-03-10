#### hetu.reduce

```
hetu.reduce(input: hetu.tensor, mode: str, axes: List[int] = [], keepdims: bool = False) -> hetu.tensor
```

Performs a reduction operation along the specified axes of the input tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.
* mode (str): The reduction type. Supports ‘sum’, ‘mean’, ‘prod’, ‘max’, ‘min’.
* axes (List[int], optional): A list of axes to reduce along. If empty, all dimensions are reduced.
* keepdims (bool, optional): Whether to keep the dimensions of the input tensor. Default is False.

**Returns:**

* hetu.tensor: A tensor with the reduction result.


