#### hetu.hardshrink

```
hetu.hardshrink(input: hetu.tensor, lambda: float = 0.5) -> hetu.tensor
```

Applies the Hard Shrink function: output = input if |input| > lambda else 0.

**Parameters:**

* input (hetu.tensor): The input tensor, which must not be empty.

* lambda (float, optional): The threshold for shrinking. Default is 0.5.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the shrunk values.

