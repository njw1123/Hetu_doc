#### hetu.interpolate

```
hetu.interpolate(input: hetu.tensor, outshape: List[int], align_corners: bool = False, scale_factor: float = 0) -> hetu.tensor
```

Performs upsampling or downsampling on a 4D input tensor, adjusting the spatial dimensions via target shape or scale factor.

**Parameters:**

* input (hetu.tensor): The input tensor with shape (N, C, H_in, W_in), where N is the batch size, C is the number of channels, H_in is the input height, and W_in is the input width.

* outshape (List[int]): The target spatial shape (H_out, W_out). If specified, the scale_factor parameter is ignored.

* align_corners (bool, optional): Whether to align corner pixels. Default is False.

  * True: Input and output corner pixels correspond precisely (best for high-precision geometric transformations).

  * False: Corner pixels may stretch or compress (best for standard interpolation).

* scale_factor (float, optional): The scaling factor for spatial dimensions. Must be greater than zero. Only used if outshape is not specified.

**Returns:**

* hetu.tensor: The output tensor with shape (N, C, H_out, W_out), with the same data type and device as the input.

