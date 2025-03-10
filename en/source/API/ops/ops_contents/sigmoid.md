#### hetu.sigmoid

```
hetu.sigmoid(input: hetu.tensor) -> hetu.tensor
```

Applies the Sigmoid activation function element-wise on the input tensor: output = 1 / (1 + exp(-input)).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the Sigmoid-transformed values.

```
hetu.sigmoid_(input: hetu.tensor) -> hetu.tensor
```

Applies the Sigmoid activation function in-place, directly modifying the input tensor: output = 1 / (1 + exp(-input)).

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

**Returns:**

* hetu.tensor: The same tensor as the input, containing the Sigmoid-transformed values.

