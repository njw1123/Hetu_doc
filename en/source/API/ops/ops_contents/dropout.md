#### hetu.dropout

```
hetu.dropout(input: hetu.tensor, keep_prob: float) -> hetu.tensor
```

Applies Dropout regularization by randomly zeroing out elements of the input tensor with probability (1 - keep_prob). This helps prevent overfitting during training.

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

* keep_prob (float): The probability of keeping each element (range (0, 1]). For example, 0.8 means 80% of the elements are kept.

**Returns:**

* hetu.tensor: A new tensor with the same shape as input, with dropout applied.

```
hetu.dropout_(input: hetu.tensor, keep_prob: float) -> hetu.tensor
```

Applies Dropout regularization in-place, modifying the input tensor directly. This can save memory but is a destructive operation.

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

* keep_prob (float): The probability of keeping each element (range (0, 1]). For example, 0.8 means 80% of the elements are kept.

**Returns:**

* hetu.tensor: The same tensor as input, but modified in-place with dropout applied.

