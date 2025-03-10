#### hetu.einsum

```
hetu.einsum(msg: str, inputs: List[hetu.tensor]) -> hetu.tensor
```

Performs tensor operations based on the Einstein summation convention, supporting complex operations such as matrix multiplication, element-wise multiplication, trace extraction, and more.

**Parameters:**

* msg (str): A string defining the operation using Einstein summation notation. For example:
  * "ij,jk->ik" represents matrix multiplication.
  * "ii->i" extracts the diagonal elements.
  * "bij,bjk->bik" represents batch matrix multiplication.

* inputs (List[hetu.tensor]): A list of input tensors. The number of tensors must match the number of input operands specified in msg.

**Returns:**

* hetu.tensor: The result of the operation, with shape determined by the output indices in msg.

