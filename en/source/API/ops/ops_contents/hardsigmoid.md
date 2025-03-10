#### hetu.hardsigmoid

```
hetu.hardsigmoid(input: hetu.tensor) -> hetu.tensor
```

Approximates the Sigmoid function using a linear transformation followed by a clamp operation for efficient computation.

**Mathematical Expression:**

​	output = clamp(input / 6 + 1/2, 0, 1)

​	clamp(x, min, max) = max(min(x, max), min)	

The clamp function limits the result of the linear transformation to the range [0, 1].

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, with its values restricted to the range [0, 1].

