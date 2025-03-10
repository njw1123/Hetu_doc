#### hetu.round

```
hetu.round(tensor: hetu.tensor) -> hetu.tensor
```

Performs element-wise rounding of the input tensor values to the nearest integer.

**Parameters:**

* tensor (hetu.tensor): The input tensor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the element-wise rounded values.


```
hetu.round_(tensor: hetu.tensor) -> hetu.tensor
```

Performs element-wise rounding of the input tensor values in-place, directly modifying the original data.

**Parameters:**

* tensor (hetu.tensor): The input tensor to be modified in place.

**Returns:**

* hetu.tensor: The modified input tensor (same object as the original).


