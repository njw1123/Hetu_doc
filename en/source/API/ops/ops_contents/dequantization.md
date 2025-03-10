#### hetu.dequantization

```
hetu.dequantization(input: hetu.tensor, absmax: hetu.tensor, dqtype: hetu.dtype, blocksize: int) -> hetu.tensor
```

**Parameters:**

* input (hetu.tensor): The quantized input tensor, which can have any shape.
* absmax (hetu.tensor): The absmax tensor, which contains the maximum value for each block during quantization. The shape is (num_blocks,).
* dqtype (hetu.dtype): The target data type for dequantization (e.g., hetu.float16, hetu.float32).
* blocksize (int): The block size used during quantization, which should match the quantization operation.

**Returns:**

* hetu.tensor:  The dequantized tensor, with the same shape as input, and the data type specified by dqtype.

