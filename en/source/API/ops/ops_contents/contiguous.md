#### hetu.contiguous

```
hetu.contiguous(input: hetu.tensor) -> hetu.tensor
```

Converts the input tensor into a contiguous memory layout, ensuring correctness for subsequent operations like reshaping.

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

**Returns:**

* hetu.tensor: If the input tensor is already contiguous, the original tensor is returned (no copy). Otherwise, a new tensor is returned with the data stored in row-major order.

