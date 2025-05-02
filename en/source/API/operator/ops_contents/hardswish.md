# hetu.hardswish

```python
hetu.hardswish(input: hetu.tensor) -> hetu.tensor
```

Applies the Hard Swish activation function, which combines linear gating and clamping for efficient computation.

**Mathematical Expression:**

$$
\begin{aligned}
output &= input * \text{clamp}(0.1667 * input + 0.5, 0, 1) \\

\text{clamp}(x, min, max) &= \max\{\min\{x, max\}, min\}
\end{aligned}
$$

**Parameters:**

* `input (hetu.tensor)`: The input tensor of any shape.

**Returns:**

* `hetu.tensor`: A new tensor with the same shape as the input, with output values in the range [0, input].
