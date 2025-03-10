#### hetu.softshrink

```
hetu.softshrink(input: hetu.tensor, lambda: float = 0.5) -> hetu.tensor
```

Applies the Soft Shrink function, which compresses input values towards zero within a threshold.

**Mathematical Expression:**

```
if input > lambda:
    output = input - lambda
elif input < -lambda:
    output = input + lambda
else:
    output = 0
```

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

* lambda (float, optional): The shrinkage threshold, controlling the width of the zeroing interval. Default is 0.5 and must satisfy lambda ≥ 0.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, where values outside the interval [-lambda, lambda] are shrunk toward zero, and values within the interval are set to zero.

