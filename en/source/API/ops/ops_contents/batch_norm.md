#### hetu.batch_norm

```
hetu.batch_norm(input: hetu.tensor, bn_scale: hetu.tensor, bn_bias: hetu.tensor, running_mean: hetu.tensor, running_var: hetu.tensor, momentum: float = 0.1, eps: float = 1e-5) -> hetu.tensor
```

Performs Batch Normalization on the input tensor by standardizing it and then applying a scale and bias.

**Behavior:**

*  Training mode: Computes the mean and variance from the current batch for normalization and updates running_mean and running_var using the specified momentum.

*  Inference mode: Uses the provided running_mean and running_var for normalization without updating them.

**Parameters:**

* input (hetu.tensor): The input tensor with shape (N, C, ...), where N is the batch size and C is the number of channels.

* bn_scale (hetu.tensor): The scale parameter tensor with shape (C,), providing a scale factor for each channel.

* bn_bias (hetu.tensor): The bias parameter tensor with shape (C,), providing a bias for each channel.

* running_mean (hetu.tensor): The running mean tensor with shape (C,), used during inference and updated during training.

* running_var (hetu.tensor): The running variance tensor with shape (C,), used during inference and updated during training.

* momentum (float, optional): The momentum coefficient used for updating the running statistics. Should be in the range [0, 1]; default is 0.1.

* eps (float, optional): A small constant added to the variance for numerical stability; default is 1e-5.

**Returns:**

* hetu.tensor: A new tensor of the same shape as input, containing the normalized values.

