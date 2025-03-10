#### hetu.swiglu

```
hetu.swiglu(input: hetu.tensor) -> hetu.tensor
```

Applies the SwiGLU (Swish-Gated Linear Unit) activation function, which combines the Swish activation and a gating mechanism to enhance model expressiveness.

**Mathematical Expression:**

```
Let input be split equally along its last dimension into two sub-tensors A and B.
output = sigmoid(A) * B
The element-wise multiplication is performed between the sigmoid activation of A and B.
```

**Parameters:**

* input (hetu.tensor): The input tensor with shape (..., 2*D), where the last dimension must be even to allow splitting into two sub-tensors A and B, each with shape (..., D).

**Returns:**

* hetu.tensor: A new tensor with shape (..., D), corresponding to the gated output.

