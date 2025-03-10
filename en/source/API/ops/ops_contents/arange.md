#### hetu.arange

```
hetu.arange(start: float, end: float, step: float) -> hetu.tensor
```

Generates a 1D tensor with evenly spaced values, similar to numpy.arange.

**Parameters:**

* start (float): The starting value (inclusive).

* end (float): The ending value (exclusive).

* step (float): The step size, which must be non-zero.

**Returns:**

* hetu.tensor: A 1D tensor with shape (ceil((end - start)/step),).

