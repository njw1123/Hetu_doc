#### hetu.layer_norm

```
hetu.layer_norm(input: hetu.tensor, ln_scale: hetu.tensor, ln_bias: hetu.tensor, normalized_shape: List[int], eps: float = 0.01) -> hetu.tensor
```

Performs Layer Normalization on the specified dimensions of the input tensor by standardizing and then applying a scale and bias.

**Note:** normalized_shape specifies the dimensions from the end of the input to be normalized (e.g., for an input of shape (N, C, H, W), if normalized_shape is (C, H, W), normalization is applied over these dimensions).

**Parameters:**

* input (hetu.tensor): The input tensor with shape (..., normalized_shape).

* ln_scale (hetu.tensor): The scale parameter tensor with a shape matching normalized_shape.

* ln_bias (hetu.tensor): The bias parameter tensor with a shape matching normalized_shape.

* normalized_shape (List[int]): The list of dimensions (from the end) to normalize.

* eps (float, optional): A small constant for numerical stability; default is 0.01.

**Returns:**

* hetu.tensor: A new tensor with the same shape as input, containing the layer-normalized values.

