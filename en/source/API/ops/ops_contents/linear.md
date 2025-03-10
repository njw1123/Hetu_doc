#### hetu.linear

```
hetu.linear(a: hetu.tensor, b: hetu.tensor, trans_a: bool = False, trans_b: bool = False) -> hetu.tensor
```

Performs a linear transformation on two input tensors by computing matrix multiplication. 

**Parameters:**

* a (hetu.tensor): The first input tensor, with shape (M, K) when trans_a=False, or (K, M) when trans_a=True.

* b (hetu.tensor): The second input tensor, with shape (K, N) when trans_b=False, or (N, K) when trans_b=True.

* trans_a (bool, optional): Whether to transpose tensor a before the operation. Default is False.

* trans_b (bool, optional): Whether to transpose tensor b before the operation. Default is False.

**Returns:**

* hetu.tensor: The result of the matrix multiplication, with shape (M, N).

```
hetu.linear(a: hetu.tensor, b: hetu.tensor, bias: hetu.tensor, trans_a: bool = False, trans_b: bool = False) -> hetu.tensor
```

Performs a linear transformation on two input tensors by computing matrix multiplication and adding a bias term.

**Parameters:**

* a (hetu.tensor): The first input tensor, with shape (M, K) when trans_a=False, or (K, M) when trans_a=True.

* b (hetu.tensor): The second input tensor, with shape (K, N) when trans_b=False, or (N, K) when trans_b=True.

* bias (hetu.tensor): The bias term, with shape compatible with the last dimension of the result tensor (typically shape (N,)).

* trans_a (bool, optional): Whether to transpose tensor a before the operation. Default is False.

* trans_b (bool, optional): Whether to transpose tensor b before the operation. Default is False.

**Returns:**

* hetu.tensor: The result of the matrix multiplication plus the bias term, with shape (M, N).

