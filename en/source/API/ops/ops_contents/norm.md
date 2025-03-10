#### hetu.norm

```
hetu.norm(input: hetu.tensor, p: int = 1, dim: int = 0, keepdim: bool = False) -> hetu.tensor
```

Computes the vector norm of the input tensor along the specified dimension, supporting common norm types.

**Parameters:**

* input (hetu.tensor): The input tensor, which can have any shape.

* p (int, optional): The type of norm. Possible values:
  * 1: L1 norm (sum of absolute values).
  * 2: L2 norm (square root of the sum of squares).
  * 0: Count of non-zero elements (only supports dim=None for global statistics).
  * Other integers: Generalized L-p norm.

* dim (int, optional): The dimension along which to compute the norm. Default is 0.

* keepdim (bool, optional) Whether to retain the reduced dimensions. Default is False.
  * True: Keeps the dim dimension in the output shape with size 1.
  * False: Removes the dim dimension from the output shape.

**Returns:**

* hetu.tensor: The output tensor with the shape of the input tensor, minus the dim dimension (if keepdim=False).

