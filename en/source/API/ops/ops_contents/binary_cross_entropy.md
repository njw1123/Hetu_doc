#### hetu.binary_cross_entropy

```
hetu.binary_cross_entropy(preds: hetu.tensor, labels: hetu.tensor, reduction: str = "mean") -> hetu.tensor
```

Computes the binary cross-entropy loss, which measures the difference between the predicted probabilities and the true binary labels.

**Parameters:**

* preds (hetu.tensor): The predicted probability tensor.

* labels (hetu.tensor): The true labels tensor, with the same shape as preds, where values are 0 or 1.

* reduction (str, optional): The method to reduce the loss. Options include:
  * "none": No reduction. Returns the loss for each sample.
  * "mean": Returns the mean of the loss across all samples.
  * "sum": Returns the sum of the loss across all samples. 
  * Default is "mean".

**Returns:**

* hetu.tensor: The computed loss, with the shape determined by the reduction argument. If "none", the shape will match labels.

