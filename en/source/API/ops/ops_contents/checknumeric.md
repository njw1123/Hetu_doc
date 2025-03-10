#### hetu.checknumeric

```
hetu.checknumeric(x: hetu.tensor) -> hetu.tensor
```

Checks for invalid numerical values (such as NaN or Inf) within the input tensor. This is useful for debugging and ensuring numerical stability in the model during training or inference.

**Parameters:**

* x (hetu.tensor): The input tensor, which can have any shape and data type. The tensor is checked for the presence of invalid numerical values (NaN, Inf).

**Returns:**

* hetu.tensor: A boolean tensor with the same shape as x. Each element will be True if the corresponding value in x is invalid (NaN or Inf), and False otherwise.

