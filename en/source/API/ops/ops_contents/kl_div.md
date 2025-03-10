#### hetu.kl_div

```
hetu.kl_div(preds: hetu.tensor, labels: hetu.tensor, reduction: str = "mean") -> hetu.tensor
```

Computes the Kullback-Leibler (KL) divergence, which measures the difference between two probability distributions.

**Parameters:**

* preds (hetu.tensor): The predicted log probabilities tensor.

* labels (hetu.tensor): The target probability distribution tensor, with the same shape as preds

* reduction (str, optional)**: The method to reduce the loss. Options include:

  * "none": No reduction. Returns the loss for each sample.

  * "mean": Returns the mean of the loss across all samples.

  * "sum": Returns the sum of the loss across all samples. 

  * Default is "mean".

**Returns:**

* hetu.tensor: The computed KL divergence, with the shape determined by the reduction argument. If "none", the shape will match labels.

