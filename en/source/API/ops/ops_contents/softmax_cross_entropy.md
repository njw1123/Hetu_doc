#### hetu.softmax_cross_entropy

```
hetu.softmax_cross_entropy(preds: hetu.tensor, labels: hetu.tensor, reduction: str = "mean") -> hetu.tensor
```

Computes the Softmax cross-entropy loss, which combines the softmax activation function with the cross-entropy loss, used for multi-class classification tasks.

**Parameters:**

* preds (hetu.tensor): The raw prediction tensor, containing unnormalized class scores.

* labels (hetu.tensor): The true labels tensor, with one fewer dimension than preds.

* reduction (str, optional)**: The method to reduce the loss. Options include:

  * "none": No reduction. Returns the loss for each sample.

  * "mean": Returns the mean of the loss across all samples.

  * "sum": Returns the sum of the loss across all samples. 

  * Default is "mean".

**Returns:**

* hetu.tensor: The computed Softmax cross-entropy loss, with the shape determined by the reduction argument.

