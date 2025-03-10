#### hetu.range_mask

```
hetu.range_mask(input: hetu.tensor, min: int, max: int) -> hetu.tensor
```

Applies a range filter to the input tensor elements, setting values outside the [min, max] range to zero, and keeping the original values for those within the range.

**Parameters:**

* input (hetu.tensor): The input tensor with any shape. The element type can be either integer or floating point.

* min (int): The minimum threshold (inclusive).

* max (int): The maximum threshold (inclusive).

**Returns:**

* hetu.tensor: The output tensor with the same shape as the input, where values in the range [min, max] are preserved, and others are set to zero.

