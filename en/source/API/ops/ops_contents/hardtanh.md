#### hetu.hardtanh

```
hetu.hardtanh(input: hetu.tensor, min_val: float = -1, max_val: float = 1) -> hetu.tensor
```

Applies the Hard Tanh function by clamping the input tensor to a specified range.

**Mathematical Expression:**

​	output = clamp(input, min_val, max_val)

​	clamp(x, min, max) = max(min(x, max), min)

The clamp function restricts the values to the interval [min_val, max_val].

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

* min_val (float, optional): The lower bound for clamping. Default is -1.

* max_val (float, optional): The upper bound for clamping. Default is 1.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, where values are clamped within [min_val, max_val].

