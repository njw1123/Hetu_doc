# hetu.exp

```python
hetu.exp(input: hetu.tensor) -> hetu.tensor
```

Computes the exponential of each element in the input tensor element-wise: output = e^input.

**Parameters:**

* `input (hetu.tensor)`: The input tensor, which must not be empty.

**Returns:**

* `hetu.tensor`: A new tensor with the same shape as the input, containing the exponential of each element.

```python
hetu.exp_(input: hetu.tensor) -> hetu.tensor
```

Applies the exponential function in-place, directly modifying the input tensor: output = e^input.

**Parameters:**

* `input (hetu.tensor)`: The input tensor, which must not be empty.

**Returns:**

* `hetu.tensor`: The same tensor as the input, containing the exponential of each element.
