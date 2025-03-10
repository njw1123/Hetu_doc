#### hetu.as_strided

```
hetu.as_strided(input: hetu.tensor, outshape: List[int], stride: List[int], storage_offset: int = 0) -> hetu.tensor
```

Creates a view of the input tensor with the specified shape and strides (no data copy), useful for advanced indexing operations.

**Parameters:**

* input (hetu.tensor): The input tensor.

* outshape (List[int]): The target shape, which must not exceed the storage capacity of the original tensor.

* stride (List[int]): The stride for each dimension, the length must match outshape.

* storage_offset (int, optional): The starting offset in storage. Default is 0.

**Returns:**

* hetu.tensor: A view tensor that shares storage with the original tensor.

