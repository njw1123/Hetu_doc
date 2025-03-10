#### hetu.pad

```
hetu.pad(input: hetu.tensor, paddings: List[int], mode: str, constant: float) -> hetu.tensor
```

Performs edge padding on the input tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.
* paddings (List[int]): Padding rules for each dimension in the tensor, formatted as [front_padding, back_padding] for each dimension.
* mode (str): The padding mode, currently only ‘constant’ is supported (constant padding).
* constant (float): The constant value to pad with, used only when mode='constant'.

**Returns:**

* hetu.tensor: A new tensor with the applied padding.


