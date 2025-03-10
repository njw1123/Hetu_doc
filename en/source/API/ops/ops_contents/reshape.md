#### hetu.reshape

```
hetu.reshape(input: hetu.tensor, output_shape: List[int]) -> hetu.tensor
```

Reshapes the input tensor to the specified dimensions. The total number of elements must be the same as in the original shape.

**Parameters:**

* input (hetu.tensor): The input tensor to be reshaped.
* output_shape (List[int]): A list specifying the target shape for the tensor. The total number of elements must be the same as the input tensor’s total number of elements.

**Returns:**

* hetu.tensor: A new tensor with the reshaped dimensions.

```
hetu.reshape(input: hetu.tensor, output_shape: List[hetu.IntSymbol]) -> hetu.tensor
```

Symbolically reshapes the tensor (used for dynamic shape computation).

**Parameters:**

* input (hetu.tensor): The input tensor to be reshaped.
* output_shape (List[hetu.IntSymbol]): A list specifying the target shape with symbolic dimensions.

**Returns:**

* hetu.tensor: A new tensor with the reshaped dimensions.


