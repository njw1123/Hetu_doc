#### hetu.avgpool

```
hetu.avgpool(input: hetu.tensor, kernel_H: int, kernel_W: int, padding: int, stride: int) -> hetu.tensor
```

Performs 2D average pooling operation, downsampling the input tensor by computing the average value in each pooling window.

**Parameters:**

* input (hetu.tensor): The input tensor, with shape (N, C, H_in, W_in).
  * N: Batch size
  * C: Number of input channels
  * H_in/W_in: Height/width of the input

* kernel_H (int): The height of the pooling window.

* kernel_W (int): The width of the pooling window.

* padding (int): Padding size for both height and width (same padding on all sides).

* stride (int): The stride of the pooling window, which is the step size for sliding the window.

**Returns:**

* hetu.tensor: The output tensor with shape (N, C, H_out, W_out), where:
  * H_out = (H_in + 2 * padding - kernel_H) // stride + 1
  * W_out = (W_in + 2 * padding - kernel_W) // stride + 1

