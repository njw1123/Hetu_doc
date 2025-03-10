#### hetu.broadcast

```
hetu.broadcast(input: hetu.tensor, shape: List[int], add_axes: List[int]) -> hetu.tensor
```

Broadcasts the input tensor to the specified shape, supporting the insertion of new dimensions at specified positions. The dimensions to be inserted should match the difference between the input tensor’s shape and the target shape. If no axes are specified, new dimensions will be inserted from the front.

**Parameters:**

* input (hetu.tensor): The input tensor.
* shape (List[int]): A list specifying the target broadcast shape.
* add_axes (List[int]): A list of axis indices where new dimensions should be inserted. 

**Returns:**

* hetu.tensor: A new tensor with the target shape.


```
hetu.broadcast(input: hetu.tensor, output: hetu.tensor) -> hetu.tensor
```

Broadcasts the input tensor to the shape of the target output tensor.

**Parameters:**

* input (hetu.tensor): The input tensor.
* output (hetu.tensor): The tensor whose shape is used as the target for broadcasting.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the output tensor.


