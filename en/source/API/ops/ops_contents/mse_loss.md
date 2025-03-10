#### hetu.mse_loss

```
hetu.mse_loss(preds: hetu.tensor, labels: hetu.tensor, reduction: str = "mean") -> hetu.tensor
```

Computes the mean squared error (MSE) loss, commonly used for regression tasks.

**Parameters:**

* preds (hetu.tensor): The predicted values tensor, with any shape.

* labels (hetu.tensor): The true values tensor, with the same shape as preds.

* reduction (str, optional)**: The method to reduce the loss. Options include:

  * "none": No reduction. Returns the loss for each sample.

  * "mean": Returns the mean of the loss across all samples.

  * "sum": Returns the sum of the loss across all samples. 

  * Default is "mean".

**Returns:**

* hetu.tensor: The computed loss, with the shape determined by the reduction argument.

