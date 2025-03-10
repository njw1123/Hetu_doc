#### hetu.sin

```
hetu.sin(input: hetu.tensor) -> hetu.tensor
```

Computes the sine of each element in the input tensor element-wise: output = sin(input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the sine of each element.

```
hetu.sin_(input: hetu.tensor) -> hetu.tensor
```

Applies the sine function in-place, directly modifying the input tensor: output = sin(input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: The same tensor as the input, containing the sine of each element.

