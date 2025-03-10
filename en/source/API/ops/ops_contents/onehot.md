#### hetu.onehot

```
hetu.onehot(input: hetu.tensor, num_classes: int) -> hetu.tensor
```

Converts an integer index tensor into a one-hot encoded tensor.

**Parameters:**

* input (hetu.tensor): The integer index tensor, where the values must satisfy 0 ≤ input[i] < num_classes.

* num_classes (int): The total number of classes, determining the size of the last dimension of the output tensor.

**Returns:**

* hetu.tensor: The output tensor with shape input.shape + (num_classes,), where the corresponding positions are set to 1, and the rest are set to 0.

