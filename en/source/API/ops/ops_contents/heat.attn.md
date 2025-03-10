#### heat.attn

```
hetu.attn(q: hetu.tensor, k: hetu.tensor, v: hetu.tensor, p_dropout: float = 0, softmax_scale: float = -1, is_causal: bool = False, return_softmax: bool = False) -> List[hetu.tensor]
```

Performs standard multi-head attention computation with the formula: Attention(Q, K, V) = softmax(QK^T / softmax_scale) * V.

**Parameters:**

* q/k/v (hetu.tensor): Query, key, and value tensors, each with shape (batch_size, seq_len, num_heads, head_dim).

* p_dropout (float, optional): The dropout probability for attention weights. Default is 0 (no dropout).

* softmax_scale (float, optional): The scaling factor for softmax. If -1, it is automatically computed as 1 / sqrt(head_dim).

* is_causal (bool, optional): Whether to apply causal masking (prevents information leakage from future tokens). Default is False.

* return_softmax (bool, optional): Whether to return the softmax weights. Default is False.

**Returns:**

* A list containing:

  * output (hetu.tensor): The attention output, with the same shape as the input.

  * softmax_lse (hetu.tensor): The log-sum-exp of the softmax values, with shape (batch_size, num_heads, seq_len), used for gradient computation.

  *  p (hetu.tensor): The attention weights, with shape (batch_size, num_heads, seq_len, seq_len).

  * rng_state (hetu.tensor): The random number generator state, with shape (2,), located on the CPU.

