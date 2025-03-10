#### hetu.matmul

```
hetu.matmul(input: hetu.tensor, other: hetu.tensor, trans_a: bool = False, trans_b: bool = False) -> hetu.tensor
```

Performs generalized matrix multiplication, supporting scalar, vector, matrix, and batch matrix multiplication. It automatically handles dimension compatibility according to the input shapes and transpose flags.

* Vector-Vector multiplication (dim_a=1 and dim_b=1): The input shapes must be the same, and the result returns a scalar.
  * Example: a.shape=(5,) × b.shape=(5,) → output.shape=()

* Vector-matrix multiplication:
  * Vector left-multiplied by matrix (dim_a=1 and dim_b=2): The length of the vector must equal the number of rows of the matrix (when trans_b=False) or the number of columns of the matrix (when trans_b=True).
    * Example: a.shape=(5,) × b.shape=(5,3) → output.shape=(3,)
  * Matrix right-multiplied by vector (dim_a=2 and dim_b=1): The number of columns in the matrix (when trans_a=False) or the number of rows in the matrix (when trans_a=True) must equal the length of the vector.
    * Example: a.shape=(2,5) × b.shape=(5,) → output.shape=(2,)
* matrix-matrix multiplication:
  * Matrix-matrix multiplication (dim_a=2 and dim_b=2): The number of columns in the first matrix must equal the number of rows in the second matrix before transposition.
    * Example: a.shape=(2,5) × b.shape=(5,3) → output.shape=(2,3)
  * Batch matrix multiplication (dim_a >= 3 or dim_b >= 3): The batch dimensions are automatically broadcasted (following NumPy broadcasting rules). The last two dimensions follow the matrix multiplication rules.
    * Example: a.shape=(10,2,5) × b.shape=(5,3) → output.shape=(10,2,3)

**Parameters:**

* input (hetu.tensor): The first input tensor

* other (hetu.tensor): The second input tensor

* trans_a (bool, optional): Whether to transpose the last two dimensions of input. Default is False. For higher-dimensional tensors, this only affects the last two dimensions.

* trans_b (bool, optional): Whether to transpose the last two dimensions of other. Default is False. For higher-dimensional tensors, this only affects the last two dimensions.

**Returns:**

* hetu.tensor: The result of the matrix multiplication, with the shape dynamically generated based on the input dimensions and transpose flags

