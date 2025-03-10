#### hetu.triu

```
hetu.triu(input: hetu.tensor, lower: bool = False, diagonal: int = 0) -> hetu.tensor
```

Generates an upper or lower triangular matrix, keeping or zeroing out the elements as required.

**Parameters:**

* input (hetu.tensor): The input tensor.

* lower (bool, optional): If True, keeps the lower triangle; otherwise, keeps the upper triangle. Default is False.

* diagonal (int, optional): The diagonal offset. Default is 0, which affects the main diagonal.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the upper or lower triangular matrix.

