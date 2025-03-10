#### hetu.where

```
hetu.where(cond: hetu.tensor, inputA: hetu.tensor, inputB: hetu.tensor) -> hetu.tensor
```

Selects elements from inputA and inputB based on the boolean condition tensor cond, similar to a ternary operation.

**Parameters:**

* cond (hetu.tensor): The boolean tensor, which must have the same shape as inputA and inputB.

* inputA (hetu.tensor): The values to select when cond is True.

* inputB (hetu.tensor): The values to select when cond is False.

**Returns:**

* hetu.tensor: The output tensor with the same shape as cond, with elements chosen based on the condition.

