# hetu.index_add_

```python
hetu.index_add_(x: hetu.tensor, y: hetu.tensor, dim: int, start_and_end_idx: List[int]) -> hetu.tensor
```

Accumulates the values from `y` into `x` at the specified indices along the specified dimension `dim`.

**Parameters:**

* `x (hetu.tensor)`: The target tensor to be modified in place.
* `y (hetu.tensor)`: The tensor containing the values to be added.
* `dim (int)`: The dimension along which to add the values. Currently we only support `dim=0`.
* `start_and_end_idx (List[int])`: A list containing the start and end indices for the addition operation.

```python
hetu.index_add_(x: hetu.tensor, y: hetu.tensor, dim: int, start_and_end_idx: HTShape) -> hetu.tensor
```

Accumulates the values from `y` into `x` at the specified indices along the specified dimension `dim`.

**Parameters:**

* `x (hetu.tensor)`: The target tensor to be modified in place.
* `y (hetu.tensor)`: The tensor containing the values to be added.
* `dim (int)`: The dimension along which to add the values. Currently we only support `dim=0`.
* `start_and_end_idx (List[int])`: A list containing the start and end indices for the addition operation.

```python
hetu.index_add_(x: hetu.tensor, y: hetu.tensor, task_batch_idx: hetu.tensor, dim: int) -> hetu.tensor
```

Accumulates the values from `y` into `x` at the specified indices along the specified dimension `dim`.

**Parameters:**

* `x (hetu.tensor)`: The target tensor to be modified in place.
* `y (hetu.tensor)`: The tensor containing the values to be added.
* `task_batch_idx (hetu.tensor)`: The tensor containing the indices for the addition operation for multiple tasks with dynamic batch size.
* `dim (int)`: The dimension along which to add the values. Currently we only support `dim=0`.
