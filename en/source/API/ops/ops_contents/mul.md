#### hetu.mul

```
hetu.mul(input: hetu.tensor, other: hetu.tensor) -> hetu.tensor
```

Performs element-wise multiplication of two tensors. The shapes of input and other must be compatible for broadcasting.

**Parameters:**

* input (hetu.tensor): The first input tensor.
* other (hetu.tensor): The second input tensor. The shape must be compatible with the input tensor for broadcasting.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensors, containing the result of the element-wise multiplication.



```
hetu.mul(tensor: hetu.tensor, value: float) -> hetu.tensor
```

Multiplies each element of the tensor by a scalar value.

**Parameters:**

* tensor (hetu.tensor): The input tensor.
* value (float): The scalar multiplier.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of multiplying each element by the scalar value.


```
hetu.mul(value: float, tensor: hetu.tensor) -> hetu.tensor
```

Multiplies a scalar value with each element of the tensor.

**Parameters:**

* value (float): The scalar multiplier.
* tensor (hetu.tensor): The input tensor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of multiplying the scalar value with each element.


