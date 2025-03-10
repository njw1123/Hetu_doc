#### hetu.conv2d

```
hetu.conv2d(input: hetu.tensor, filter: hetu.tensor, padding: int, stride: int) -> hetu.tensor
```

Performs a 2D convolution operation, applying a filter on the input tensor for feature extraction.

**Parameters:**

* input (hetu.tensor): The input tensor, with shape (N, C_in, H_in, W_in), where:
  * N: Batch size
  * C_in: Number of input channels
  * H_in/W_in: Height/width of the input

* filter (hetu.tensor): The convolution filter tensor, with shape (C_out, C_in, kH, kW), where:

  * C_out: Number of output channels

  * kH/kW: Height/width of the filter

* padding (int): The padding size for both height and width (same padding on all sides).

* stride (int): The stride of the convolution filter, which is the step size for sliding the filter in both height and width directions.

**Returns:**

* hetu.tensor: The output tensor with shape (N, C_out, H_out, W_out), where:

  * H_out = (H_in + 2 * padding - kH) // stride + 1

  * W_out = (W_in + 2 * padding - kW) // stride + 1

```
hetu.conv2d(input: hetu.tensor, filter: hetu.tensor, bias: hetu.tensor, padding: int, stride: int) -> hetu.tensor
```

Performs a 2D convolution operation with bias, applying a filter on the input tensor and adding a bias term to the output.

**Parameters:**

* input (hetu.tensor): Same as the previous version, with shape (N, C_in, H_in, W_in).

* filter (hetu.tensor): The convolution filter tensor, with shape (C_out, C_in, kH, kW).

* bias (hetu.tensor): The bias term, with shape (C_out,). Each output channel corresponds to one bias value.

* padding (int): Padding size for both height and width.

* stride (int): The stride of the convolution filter.

**Returns:**

* hetu.tensor: The output tensor with shape (N, C_out, H_out, W_out), with the bias added to the result.


