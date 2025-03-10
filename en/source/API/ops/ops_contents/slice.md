#### hetu.slice

```
hetu.slice(input: hetu.tensor, begin_pos: List[int], output_shape: List[int]) -> hetu.tensor
```

Slices the input tensor starting from the specified positions to create a sub-tensor with the specified shape.

**Parameters:**

* input (hetu.tensor): The input tensor.
* begin_pos (List[int]): A list specifying the starting position for slicing along each dimension. The length must be equal to the number of dimensions in the input tensor.
* output_shape (List[int]): The shape of the resulting sub-tensor. The length must match the number of dimensions in the input tensor.

**Returns:**

* hetu.tensor: A new tensor containing the sliced sub-tensor.

```
hetu.slice(input: hetu.tensor, begin_pos: List[hetu.IntSymbol], output_shape: List[hetu.IntSymbol]) -> hetu.tensor
```

Symbolically slices the input tensor for dynamic shape computation.

**Parameters:**

- input (hetu.tensor): The input tensor.

* begin_pos (List[hetu.IntSymbol]): A list of symbolic positions to start slicing from.

* output_shape (List[hetu.IntSymbol]): A list of symbolic dimensions for the sliced sub-tensor.

**Returns:**

* hetu.tensor: A new tensor containing the symbolically sliced sub-tensor.

