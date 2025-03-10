#### hetu.sub

```
hetu.sub(input: hetu.tensor, other: hetu.tensor) -> hetu.tensor
```

Performs element-wise subtraction of one tensor from another. The shapes of input and other must be compatible for broadcasting.

**Parameters:**

* input (hetu.tensor): The tensor to subtract from (minuend).
* other (hetu.tensor): The tensor to subtract (subtrahend). The shape must be compatible with the input tensor for broadcasting.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensors, containing the result of the element-wise subtraction.


```
hetu.sub(tensor: hetu.tensor, value: float) -> hetu.tensor
```

Subtracts a scalar value from each element of the input tensor.

**Parameters:**

* tensor (hetu.tensor): The input tensor.
* value (float): The scalar value to subtract from each element of the tensor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of subtracting the scalar value.


```
hetu.sub(value: float, tensor: hetu.tensor) -> hetu.tensor
```

Subtracts each element of the tensor from the scalar value.

**Parameters:**

* value (float): The scalar value from which each element of the tensor is subtracted.
* tensor (hetu.tensor): The input tensor whose elements will be subtracted from the scalar.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of the subtraction.

