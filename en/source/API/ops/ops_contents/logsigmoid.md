#### hetu.logsigmoid

```
hetu.logsigmoid(input: hetu.tensor) -> hetu.tensor
```

**Mathematical Expression:**

​	output = log(1 / (1 + exp(-input)))

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the log-sigmoid values.

