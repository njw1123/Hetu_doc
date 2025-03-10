#### hetu.bmm

```
hetu.bmm(input: hetu.tensor, other: hetu.tensor, trans_a: bool = False, trans_b: bool = False) -> hetu.tensor
```

Performs batch matrix multiplication. The inputs must be 3D tensors.

**Parameters:**

* input (hetu.tensor): The first input tensor, with shape (batch_size, M, K) when trans_b=False, or (batch_size, N, K) when trans_b=True.

* other (hetu.tensor): The second input tensor, with shape (batch_size, K, N) when trans_b=False, or (batch_size, N, K) when trans_b=True.

* trans_a (bool, optional): Whether to transpose the last two dimensions of input. Default is False.

* trans_b (bool, optional): Whether to transpose the last two dimensions of other. Default is False.

**Returns:**

* hetu.tensor: The result of batch matrix multiplication, with shape (batch_size, M, N).

