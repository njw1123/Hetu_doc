#### hetu.transpose

```
hetu.transpose(input: hetu.tensor, perms: List[int]) -> hetu.tensor
```

Transposes the input tensor by permuting its dimensions.

**Parameters:**

* input (hetu.tensor): The input tensor to be transposed.

* perms (List[int]): A list specifying the new order of dimensions (e.g., [2, 3, 1, 0] for reshaping [N,C,H,W] to [H,W,C,N]).

**Returns:**

* hetu.tensor: A new tensor with its dimensions permuted according to the specified order.

