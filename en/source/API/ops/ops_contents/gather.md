#### hetu.gather

```
hetu.gather(input: hetu.tensor, dim: int, id: hetu.tensor) -> hetu.tensor
```

Gathers elements from the input tensor along the specified dimension.

For a 3D tensor, the output is specified by:

```
	out[i][j][k] = input[index[i][j][k]][j][k] if dim == 0
	out[i][j][k] = input[i][index[i][j][k]][k] if dim == 1
	out[i][j][k] = input[i][j][index[i][j][k]] if dim == 2
```

**Parameters:**

* input (hetu.tensor): The input tensor.

* dim (int): The dimension along which to gather. Must satisfy 0 <= dim < input.ndim.

* id (hetu.tensor): The index tensor. The shape of the non-dim dimensions must match the input, and the indices must be within the range [0, input.shape(dim)).

**Returns:**

* hetu.tensor: The gathered tensor with shape id.shape.

