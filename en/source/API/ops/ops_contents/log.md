#### hetu.log

```
hetu.log(input: hetu.tensor) -> hetu.tensor
```

Computes the natural logarithm of each element in the input tensor element-wise: output = ln(input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the natural logarithm of each element.

```
hetu.log_(input: hetu.tensor) -> hetu.tensor
```

Applies the natural logarithm function in-place, directly modifying the input tensor: output = ln(input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: The same tensor as the input, containing the natural logarithm of each element.

