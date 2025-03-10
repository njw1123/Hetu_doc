#### hetu.nll_loss

```
hetu.nll_loss(preds: hetu.tensor, labels: hetu.tensor, reduction: str = "mean") -> hetu.tensor
```

Computes the negative log likelihood loss, used for classification tasks where the input is the log of predicted probabilities.

**Parameters:**

* preds (hetu.tensor): The predicted log probabilities tensor.

* labels (hetu.tensor): The true labels tensor, which should have one fewer dimension than preds.

* reduction (str, optional)**: The method to reduce the loss. Options include:
  * "none": No reduction. Returns the loss for each sample.
  * "mean": Returns the mean of the loss across all samples.
  * "sum": Returns the sum of the loss across all samples. 
  * Default is "mean".

**Returns:**

* hetu.tensor: The computed loss, with the shape determined by the reduction argument. If "none", the shape will match labels.

