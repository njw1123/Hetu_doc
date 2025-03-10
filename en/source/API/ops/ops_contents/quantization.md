#### hetu.quantization

```
hetu.quantization(input: hetu.tensor, qtype: hetu.dtype, blocksize: int, stochastic: bool = False) -> hetu.tensor
```

Performs block quantization on the input tensor, converting floating-point data into a low-precision integer representation. This is useful for model compression and inference acceleration.

**Parameters:**

* input (hetu.tensor): The input tensor, which can have any shape. The data type must be float32 or float16.
* qtype (hetu.dtype): The target quantization data type. It could be an integer type like int8 or int16, depending on the compression needs.
* blocksize (int): The block size for quantization, which must be a power of 2 (e.g., 64, 128).
* stochastic (bool, optional): Whether to use stochastic rounding, which reduces quantization bias. Default is False.

**Returns:**

* hetu.tensor: The quantized tensor, with the same shape as the input tensor. The data type is qtype. Internally, every block of size blocksize will have a scaling factor (absmax tensor) appended.

