#### hetu.leakyrelu

```
hetu.leakyrelu(input: hetu.tensor, alpha: float = 0.01) -> hetu.tensor
```

Applies the Leaky ReLU activation function element-wise on the input tensor: output = max(alpha * input, input), where alpha is the slope for negative values.

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

* alpha (float, optional): The slope for the negative input values. Default is 0.01.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, where the negative values are scaled by alpha.

```
hetu.leakyrelu_(input: hetu.tensor, alpha: float = 0.01) -> hetu.tensor
```

Applies the Leaky ReLU activation function in-place, directly modifying the input tensor: output = max(alpha * input, input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

* alpha (float, optional): The slope for the negative input values. Default is 0.01.

**Returns:**

* hetu.tensor:  The same tensor as the input, where the negative values are scaled by alpha.

