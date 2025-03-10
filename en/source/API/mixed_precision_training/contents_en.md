### MixedPrecisionTraining

#### hetu.autocast
The autocast context manager allows for mixed precision training, where you can define the data type used during training, such as hetu.bfloat16

```
with hetu.autocast(hetu.dtype):
	# Training code
```

**Parameters**

* hetu.dtype: The desired data type for the training process within the autocast context.  Common data types include hetu.float32, hetu.bfloat16, and others supported by Hetu.
