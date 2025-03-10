#### hetu.diagonal

```
hetu.diagonal(input: hetu.tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> hetu.tensor
```

Returns the diagonal elements from the input tensor along the specified dimensions.

**Parameters:**

* input (hetu.tensor): The input tensor, which must have at least two dimensions.

* offset (int, optional): The diagonal offset to extract. Default is 0.
  * offset = 0: Main diagonal.
  * offset > 0: Diagonals above the main diagonal.
  * offset < 0: Diagonals below the main diagonal.

* dim1 (int, optional): The first dimension to consider for extracting the diagonal. Default is 0.

* dim2 (int, optional): The second dimension to consider for extracting the diagonal. Default is 1.

**Returns:**

* hetu.tensor: A new tensor containing the extracted diagonal elements. The shape of the new tensor will be one dimension less than the original tensor, as the diagonal elements are flattened.

