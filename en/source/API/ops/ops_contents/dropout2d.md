#### hetu.dropout2d

```
hetu.dropout2d(input: hetu.tensor, keep_prob: float) -> hetu.tensor
```

Applies 2D Dropout to convolutional feature maps by randomly zeroing out entire channels, which helps regularize the model during training.

**Parameters:**

* input (hetu.tensor): A 4D input tensor with shape (N, C, H, W).

* keep_prob (float): The probability of keeping each channel (range (0, 1]).

**Returns:**

* hetu.tensor: A new tensor with the same shape as input, with dropout applied to entire channels.

```
hetu.dropout2d_(input: hetu.tensor, keep_prob: float) -> hetu.tensor
```

Applies 2D Dropout in-place on convolutional feature maps, zeroing out entire channels directly in the input tensor.

**Parameters:**

* input (hetu.tensor): A 4D input tensor with shape (N, C, H, W).

* keep_prob (float): The probability of keeping each channel (range (0, 1]).

**Returns:**

* hetu.tensor:  The same tensor as input, modified in-place with 2D dropout applied.

