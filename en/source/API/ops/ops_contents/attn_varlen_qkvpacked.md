#### hetu.attn_varlen_qkvpacked

Processes attention computation for variable-length sequences with QKV packed

* qkv (hetu.tensor): The merged Q/K/V tensor with shape (total_tokens, 3, num_heads, head_dim)

* cu_seqlens_q/k (hetu.tensor): Cumulative sequence length tensors for queries and keys, with shape (batch_size + 1,), and data type int32. Example: cu_seqlens_q = [0, 5, 8] means the first sequence is of length 5, the second sequence is of length 3.

* max_seqlen_q/k (hetu.IntSymbol): The maximum sequence length for queries and keys, represented as a symbol.

* zero_tensors (bool, optional): Whether to zero out the tensors before computation to prevent uninitialized memory issues. Default is False.

* p_dropout (float, optional): Dropout probability for attention weights, within the range [0, 1). Default is 0, meaning no dropout is applied.

* softmax_scale (float, optional): Scaling factor for the softmax computation. If set to -1, it is automatically calculated as 1 / sqrt(head_dim).

* is_causal (bool, optional): Whether to apply causal masking to prevent future tokens from attending to earlier ones. Default is False.

* return_softmax (bool, optional):Whether to return the intermediate softmax results. Default is False.

**Returns:**

* A list of four tensors:
  * output (hetu.tensor): The attention output, with the same shape as the input tensors.
  * softmax_lse (hetu.tensor): The log-sum-exp of the softmax values, with shape (batch_size, num_heads, max_seqlen_q). This is used for gradient computation.
  * p (hetu.tensor): The padded attention weights, with shape (batch_size, num_heads, max_seqlen_q_padded, max_seqlen_k_padded).
  * rng_state (hetu.tensor): The state of the random number generator, with shape (2,), located on the CPU.

