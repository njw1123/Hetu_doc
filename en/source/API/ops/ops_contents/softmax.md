#### hetu.softmax

```
hetu.softmax(input: hetu.tensor, dim: int = 0) -> hetu.tensor
```

Applies the Softmax function along the specified dimension: output = exp(input) / sum(exp(input), dim=dim).

**Parameters:**

* input (hetu.tensor): The input tensor.

* dim (int, optional): The dimension along which to apply the Softmax function. Default is 0.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the Softmax-normalized values.

