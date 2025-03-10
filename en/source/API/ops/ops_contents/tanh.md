#### hetu.tanh

```
hetu.tanh(input: hetu.tensor) -> hetu.tensor
```

Applies the hyperbolic tangent (tanh) activation function element-wise: output = tanh(input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the tanh-transformed values.

```
hetu.tanh_(input: hetu.tensor) -> hetu.tensor
```

Applies the tanh activation function in-place, modifying the input tensor directly: output = tanh(input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: The same tensor as the input, containing the tanh-transformed values.

