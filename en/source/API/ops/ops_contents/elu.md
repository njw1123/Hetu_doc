#### hetu.elu

```
hetu.elu(input: hetu.tensor, alpha: float = 1, scale: float = 1) -> hetu.tensor
```

Applies the Exponential Linear Unit (ELU) activation function:

output = scale * (max(0, input) + min(0, alpha * (exp(input) - 1))).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

* alpha (float, optional): The exponent coefficient for negative input values. Default is 1.

* scale (float, optional): The scaling factor for the output. Default is 1.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the ELU-transformed values.

