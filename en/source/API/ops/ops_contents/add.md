#### hetu.add

``` 
hetu.add(input: hetu.tensor, other: hetu.tensor) -> hetu.tensor
```

Performs element-wise addition of two tensors. The shapes of input and other must be compatible for broadcasting.

**Parameters:**

* input (hetu.tensor): The first input tensor.
* other (hetu.tensor): The second input tensor. The shape must be compatible with the input tensor for broadcasting.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensors, containing the result of the element-wise addition.


```
hetu.add(input: hetu.tensor, value: float) -> hetu.tensor 
hetu.add(value: float, input: hetu.tensor) -> hetu.tensor
```

Adds a scalar value to each element of the input tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.
* value (float): The scalar value to add to each element of the tensor.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input tensor, containing the result of adding the scalar value.


```
hetu.add_(input: hetu.tensor, other: hetu.tensor) -> hetu.tensor
```

Performs element-wise addition of two tensors in-place, directly modifying the first input tensor.

**Parameters:**

* input (hetu.tensor): The target tensor to be modified in place.
* other (hetu.tensor): The tensor to be added to the input tensor.

**Returns:**

* hetu.tensor: The modified input tensor (same object as the original).

```
hetu.add_(tensor: hetu.tensor, value: float) -> hetu.tensor
hetu.add_(value: float, tensor: hetu.tensor) -> hetu.tensor
```

Adds a scalar value to each element of the tensor in-place.

**Parameters:**

* tensor (hetu.tensor): The target tensor to be modified in place.
* value (float): The scalar value to be added to each element.

**Returns:**

* hetu.tensor: The modified tensor (same object as the original).

    

