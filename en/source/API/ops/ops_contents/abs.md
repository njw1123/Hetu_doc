#### hetu.abs

```
hetu.abs(input: hetu.tensor) -> hetu.tensor
```

Computes the absolute value of each element in the input tensor element-wise: output = |input|.

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the absolute values of each element.

```
hetu.abs_(input: hetu.tensor) -> hetu.tensor
```

Applies the absolute value function in-place, directly modifying the input tensor: output = |input|.

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: The same tensor as the input, containing the absolute values of each element.

