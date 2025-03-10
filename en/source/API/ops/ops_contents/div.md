#### hetu.div

```
hetu.div(input: hetu.tensor, other: hetu.tensor) -> hetu.tensor
```

Performs element-wise division of two tensors. The shapes of input and other must be compatible for broadcasting.

**Parameters:**

* input (hetu.tensor): The dividend tensor.
* other (hetu.tensor): The divisor tensor. The shape must be compatible with the input tensor for broadcasting.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensors, containing the result of the element-wise division.


```
hetu.div(tensor: hetu.tensor, value: float) -> hetu.tensor
```

Divides each element of the tensor by a scalar value.

**Parameters:**

* tensor (hetu.tensor): The input tensor.
* value (float): The scalar divisor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of dividing each element by the scalar value.



```
hetu.div(value: float, tensor: hetu.tensor) -> hetu.tensor
```

Divides a scalar value by each element of the input tensor.

**Parameters:**

* value (float): The scalar dividend.
* tensor (hetu.tensor): The divisor tensor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of dividing the scalar value by each element of the tensor.




