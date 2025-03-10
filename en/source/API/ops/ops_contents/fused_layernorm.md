#### hetu.fused_layernorm

```
hetu.fused_layernorm(input: hetu.tensor, ln_scale: hetu.tensor, ln_bias: hetu.tensor, normalized_shape: List[int], eps: float = 0.01, inplace: bool = False) -> hetu.tensor
```

Performs a fused version of Layer Normalization to optimize computation efficiency. It standardizes the input over the specified dimensions and applies a scale and bias, optionally performing the operation in-place.

**Note:** normalized_shape specifies the dimensions from the end of the input to be normalized (e.g., for an input of shape (N, C, H, W), if normalized_shape is (C, H, W), normalization is applied over these dimensions).

**Parameters:**

* input (hetu.tensor): The input tensor with shape (..., normalized_shape).

* ln_scale (hetu.tensor): The scale parameter tensor with a shape matching normalized_shape.

* ln_bias (hetu.tensor): The bias parameter tensor with a shape matching normalized_shape.

* normalized_shape (List[int]): The list of dimensions (from the end) to normalize.

* eps (float, optional): A small constant for numerical stability; default is 0.01.

* inplace (bool, optional): Whether to perform the operation in-place (modifying the input tensor directly). Default is False.

**Returns:**

* hetu.tensor: A tensor with the same shape as input, containing the fused layer-normalized values.

