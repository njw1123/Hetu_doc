#### hetu.silu

```
hetu.silu(input: hetu.tensor) -> hetu.tensor
```

Applies the Sigmoid-weighted Linear Unit (Swish) activation function, which multiplies the input by its sigmoid.

**Mathematical Expression:**

​	output = input * sigmoid(input)

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the Silu-transformed values.

