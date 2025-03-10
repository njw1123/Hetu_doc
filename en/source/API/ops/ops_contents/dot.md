#### hetu.dot

```
hetu.dot(a: hetu.tensor, b: hetu.tensor, axes: int = 0) -> hetu.tensor
```

Performs dot product operation along the specified axis of two tensors.

**Parameters:**

* a (hetu.tensor): The first input tensor.

* b (hetu.tensor): The second input tensor.

* axes (int, optional): Specifies the axis along which to compute the dot product. Default is 0.

**Returns:**

* hetu.tensor: The result of the dot product operation, with shape determined by the non-reduced dimensions.

