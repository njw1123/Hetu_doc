#### hetu.repeat

```
hetu.repeat(input: hetu.tensor, repeats: List[int]) -> hetu.tensor
```

Repeats the elements of the input tensor along each dimension, creating a new expanded tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.

* repeats (List[int]): A list of the number of repetitions along each dimension. The length of repeats must be greater than or equal to the number of dimensions of the input tensor. If repeats has more dimensions than the input tensor, it will automatically prepend 1 to the input shape.

**Returns:**

* hetu.tensor: The output tensor with shape determined by the input shape multiplied by the corresponding values in repeats.

