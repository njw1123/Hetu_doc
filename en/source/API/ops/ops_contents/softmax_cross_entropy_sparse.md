#### hetu.softmax_cross_entropy_sparse

```
hetu.softmax_cross_entropy_sparse(preds: hetu.tensor, labels: hetu.tensor, ignored_index: int = -1, reduction: str = "mean") -> hetu.tensor
```

Computes the sparse Softmax cross-entropy loss, which supports ignoring specific label indices.

**Parameters:**

* preds (hetu.tensor): The raw prediction tensor, containing unnormalized class scores.

* labels (hetu.tensor): The true labels tensor, with one fewer dimension than preds.

* ignored_index (int, optional): The label index to ignore. Default is -1. If this value exists in the labels, the corresponding loss is not included in the final result.

* reduction (str, optional): The method to reduce the loss. Options include:

  * "none": No reduction. Returns the loss for each sample.

  * "mean": Returns the mean of the loss across all samples.

  * "sum": Returns the sum of the loss across all samples. 

  * Default is "mean".

**Returns:**

* hetu.tensor: The computed sparse Softmax cross-entropy loss, with the shape determined by the reduction argument.

