#### hetu.instance_norm

```
hetu.instance_norm(input: hetu.tensor, eps: float = 1e-7) -> hetu.tensor
```

Performs Instance Normalization, normalizing each sample and each channel independently. This is especially useful in style transfer and similar tasks.

**Parameters:**

* input (hetu.tensor): The input tensor with shape (N, C, ...).

* eps (float, optional): A small constant added for numerical stability; default is 1e-7.

**Returns:**

* hetu.tensor: A new tensor of the same shape as input, containing the instance-normalized values.

