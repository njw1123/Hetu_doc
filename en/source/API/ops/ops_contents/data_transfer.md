#### hetu.data_transfer

```
hetu.data_transfer(input: hetu.tensor, datatype: hetu.dtype, dev: hetu.device) -> hetu.tensor
```

Transfers the input tensor to a specified device and changes its data type.

**Parameters:**

* datatype (hetu.dtype): The target data type, such as hetu.float32, hetu.int8, etc.

* dev (hetu.device): The target device for the tensor.

**Returns:**

* hetu.tensor: A new tensor, with its data stored on the target device and in the specified data type.

