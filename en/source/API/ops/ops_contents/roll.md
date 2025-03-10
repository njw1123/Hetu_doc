#### hetu.roll

```
hetu.roll(input: hetu.tensor, shifts: tuple, dims: tuple) -> hetu.tensor
```

Rolls the tensor elements along the specified dimensions, wrapping the elements around when they go out of bounds.

**Parameters:**

* input (hetu.tensor): The input tensor.

* shifts (tuple): The number of shifts for each dimension. Positive values roll the tensor to the right or down.

* dims (tuple): The dimensions along which to roll. The length of dims must match the length of shifts.

**Returns:**

* hetu.tensor: The output tensor with the same shape as the input, where the elements are rolled according to the given rules.

