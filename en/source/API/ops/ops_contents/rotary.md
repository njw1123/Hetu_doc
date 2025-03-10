#### hetu.rotary

```
hetu.rotary(x: hetu.tensor, cos: hetu.tensor, sin: hetu.tensor, interleaved: bool = False, inplace: bool = False) -> hetu.tensor
```

Applies Rotary Position Embedding (RoPE) to the input tensor, enhancing the model’s ability to capture the positional relationships of sequences. 

**Parameters:**

* x (hetu.tensor): The input tensor with the shape (batch_size, seq_len, num_heads, head_dim)

* cos/sin (hetu.tensor): Precomputed cosine and sine position encoding tensors, both with shape (seq_len, head_dim). These tensors must have the same data type as x and represent the position encoding used for the rotary operation.
* interleaved (bool, optional): Whether to use the interleaved rotation mode (as seen in models like PaLM). Default is False.
  * True: The head_dim is split into two groups, and the rotation alternates between these groups.
  * False: Standard rotation mode is used where all heads rotate simultaneously.

* inplace (bool, optional): Whether to modify the input tensor x in place. 

**Returns:**

* hetu.tensor: The rotated tensor, with the same shape as x, where positional encodings have been applied using RoPE.

