#### hetu.parallel_attn

```
hetu.parallel_attn(qkv: hetu.tensor, head_dim: int, group_query_ratio: int, multi_seq_lens_symbol: List[List[hetu.IntSymbol]], multi_cp_group_symbol: List[List[hetu.IntSymbol]], packing: bool = False, cu_seqlens_q: hetu.tensor = None, cu_seqlens_k: hetu.tensor = None, max_seqlen_q: hetu.IntSymbol = None, max_seqlen_k: hetu.IntSymbol = None, p_dropout: float = 0, softmax_scale: float = -1, is_causal: bool = True, return_softmax: bool = False) -> List[hetu.tensor]
```

Performs distributed multi-head attention computation with support for grouped queries and variable sequence lengths, suitable for large-scale model training.

**Parameters:**

* qkv (hetu.tensor): Merged Q/K/V tensor with shape (batch_size * seq_len, 3*hidden_dim), where hidden_dim = num_heads * head_dim，where num_heads is the number of attention heads, and head_dim is the dimension of each head.

* head_dim (int): The dimension of each attention head. Must satisfy hidden_dim % head_dim == 0.

* group_query_ratio (int): The ratio of groups for query heads. For example, 4 means every 4 query heads share one key-value head.

* multi_seq_lens_symbol (List[List[hetu.IntSymbol]]): A list of symbolized sequence lengths. The outer list corresponds to different mixed-parallel strategies (since hetu supports the switching between strategies), and the inner list represents the sequence lengths across different pipelines in each strategy.

* multi_cp_group_symbol (List[List[hetu.IntSymbol]]): A list of symbolized context groups. The outer list represents different mixed-parallel strategies, while the inner lists specify the partitioning of context groups (e.g., [0, 0, 0, 1, 1, 2, 2, 2], where elements with the same index belong to the same context group).

* packing (bool, optional): Whether to use packing. Default is False.

* cu_seqlens_q/k (hetu.tensor, optional): Cumulative sequence length tensors for query and key. Shape is (batch_size+1,), e.g., [0, 5, 10] indicates the first sequence has length 5, and the second has length 5.

* max_seqlen_q/k (hetu.IntSymbol, optional): The maximum sequence length in the batch.

* p_dropout (float, optional): The dropout probability for attention weights. Default is 0 (no dropout).

* softmax_scale (float, optional): The scaling factor for softmax. If -1, it is automatically computed as 1 / sqrt(head_dim).

* is_causal (bool, optional): Whether to apply causal masking (prevents information leakage from future tokens). Default is True.

* return_softmax (bool, optional): Whether to return the softmax weights. Default is False.

**Returns:**

* A list containing:
  * output (hetu.tensor): The attention output with the same shape as the input.
  * softmax_lse (hetu.tensor): The log-sum-exp of the softmax values, with shape (batch_size, num_heads, seq_len), used for gradient computation.
  * rng_state (hetu.tensor): The random number generator state, with shape (2,), located on the CPU.



