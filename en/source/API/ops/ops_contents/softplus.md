#### hetu.softplus

```
hetu.softplus(input: hetu.tensor, beta: float = 1, threshold: float = 20) -> hetu.tensor
```

Applies the Softplus activation function, a smooth approximation to ReLU that is differentiable and numerically stable.

**Mathematical Expression:**

```
if beta * input ≤ threshold:
    output = (1 / beta) * log(1 + exp(beta * input))
else:
    output = input  (to avoid numerical overflow)
```

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

* beta (float, optional): A scaling factor controlling the sharpness of the transition. Default is 1.

* threshold (float, optional): The threshold for switching to a linear computation to prevent overflow. Default is 20.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, with values in the range (0, +∞).

