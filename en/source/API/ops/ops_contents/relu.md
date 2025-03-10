#### hetu.relu

```
hetu.relu(input: hetu.tensor) -> hetu.tensor
```

Applies the ReLU (Rectified Linear Unit) activation function element-wise on the input tensor: output = max(0, input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, where all negative values are replaced with zero.

```
hetu.relu_(input: hetu.tensor) -> hetu.tensor
```

Applies the ReLU activation function in-place, directly modifying the input tensor: output = max(0, input).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor:  The same tensor as the input, where all negative values are replaced with zero.

