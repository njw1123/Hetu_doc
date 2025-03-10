#### hetu.masked_fill

```
hetu.masked_fill(input: hetu.tensor, mask: hetu.tensor, val: float = 0) -> hetu.tensor
```

Fills the positions of the input tensor specified by a boolean mask with a specified value, leaving the other positions unchanged.

**Parameters:**

* input (hetu.tensor): The input tensor with any shape.

* mask (hetu.tensor): A boolean mask tensor that must have the same shape as the input.

* val (float, optional): The value to fill in places where the mask is True. Default is 0. The value will be automatically cast to the data type of the input tensor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, where the values are replaced by val at the positions where the mask is True.

