#### hetu.mish

```
hetu.mish(input: hetu.tensor) -> hetu.tensor
```

Applies the Mish activation function, which combines self-gating with smooth non-linearity to improve model performance.

**Mathematical Expression:**

​	output = input * tanh(softplus(input))
​	softplus(input) = log(1 + exp(input))

**Parameters:**

* input (hetu.tensor): The input tensor of any shape.

**Returns:**

* hetu.tensor: A new tensor with the same shape as the input, containing the Mish-transformed values.

