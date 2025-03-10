#### hetu.hardswish

```
hetu.hardswish(input: hetu.tensor) -> hetu.tensor
```

Applies the Hard Swish activation function, which combines linear gating and clamping for efficient computation.

**Mathematical Expression:**

​	output = input * clamp(0.1667 * input + 0.5, 0, 1)

​	clamp(x, min, max) = max(min(x, max), min)

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, with output values in the range [0, input].

