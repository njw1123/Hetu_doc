
# Models

Hetu provides GPT and Llama models. The models are implemented in a modular way, allowing for easy customization and extension.

## GPT

### GPTConfig

```python
class GPTConfig(
  self,
  vocab_size=50257,
  n_positions=1024,
  n_embd=768,
  n_layer=12,
  n_head=12,
  n_inner=None,
  activation_function="relu",
  resid_pdrop=0.1,
  embd_pdrop=0.1,
  attn_pdrop=0.1,
  layer_norm_epsilon=1e-5,
  initializer_range=0.02,
  summary_type="cls_index",
  summary_use_proj=True,
  summary_activation=None,
  summary_proj_to_labels=True,
  summary_first_dropout=0.1,
  scale_attn_weights=True,
  use_cache=False,
  bos_token_id=50256,
  eos_token_id=50256,
  scale_attn_by_inverse_layer_idx=False,
  reorder_and_upcast_attn=False,
  use_flash_attn = True,
)
```

GPTConfig is the configuration class for GPT, defining various hyperparameters of the model.

**Parameters**:

- `vocab_size`: The size of the vocabulary.
- `n_positions`: The maximum sequence length that this model might ever be used with.
- `n_embd`: The dimension of the embedding layer and hidden states (i.e., the dimensionality of the representation vector for each token).
- `n_layer`: The number of layers in the Transformer network.
- `n_head`: The number of self-attention heads in each Transformer layer.
- `n_inner`: If set, it represents the dimensionality of the inner layer of the feed-forward network, typically larger than n_embd.
- `activation_function`: The type of activation function.
- `resid_pdrop`: The dropout ratio for residual connections.
- `embd_pdrop`: The dropout ratio for the embedding layer.
- `attn_pdrop`: The dropout ratio for the self-attention mechanism.
- `layer_norm_epsilon`: The epsilon value used in layer normalization to prevent division by zero errors.
- `initializer_range`: The range of weight initialization, used to initialize the model parameters.
- `summary_type`: The type of summary output during model training. Common values include cls_index (used for classification tasks), specifying how the model generates task-related summaries.
- `summary_use_proj`: Whether to use an additional linear transformation to project the model output into the label space. If set to True, an extra projection layer is applied.
- `summary_activation`: The activation function used for summary output, typically used in classification tasks.
- `summary_proj_to_labels`: Whether to project the summary output to the label space. If set to True, the model maps the final output layer to the label space.
- `summary_first_dropout`: The dropout ratio used when generating summaries.
- `scale_attn_weights`: Whether to scale attention weights.
- `use_cache`: Whether to enable caching.
- `bos_token_id`: The ID of the bos_token (begin-of-sequence token), representing the start of a sequence.
- `eos_token_id`: The ID of the eos_token (end-of-sequence token), representing the end of a sequence.
- `scale_attn_by_inverse_layer_idx`: Whether to scale attention weights inversely by layer index.
- `reorder_and_upcast_attn`: Whether to reorder and upcast precision during attention computation.
- `use_flash_attn`: Whether to enable FlashAttention acceleration.

### GPTModel

```python
GPTLMHeadModel(config, ds_parallel_configs) -> ht.nn.Module
```

Generates a GPT model based on the given model configuration and distributed parallel settings.

**Parameters**:

- `config (GPTConfig)`: An instance of the GPTConfig that defines the model's architecture and hyperparameters.
- `ds_parallel_configs (List)`: A list of distributed parallel configs.

### GPT ds config

```python
def generate_gpt_4d_config(num_layers, num_gpus, dp, cp, tp, pp, zero, recompute_granularity, recompute_method, recompute_num_layers, recompute_layer_idxs_list) -> Dict
```

Generates a ds_config dict based on the model architecture and parallel strategy.

**Parameters**:

- `num_layers (int, default=32)`: The total number of layers in the model.
- `num_gpus (int, default=8)`: The number of GPUs used for training.
- `dp (int, default=2)`: The degree of data parallelism (DP).
- `cp (int, default=1)`: The degree of context parallelism (CP).
- `tp (int, default=2)`: The degree of tensor parallelism (TP).
- `pp (int, default=2)`: The degree of pipeline parallelism (PP).
- `zero (bool, default=True)`: Whether to enable ZeRO-1 optimization.
- `recompute_granularity (Union[str, List[str]], optional)`: The granularity of recomputation that controls the scope of activations to recompute within a Transformer block. It can be a string or a list of strings for heterogeneous recomputation. Recompute granularity can be 'selective' and 'full'.
- `recompute_method (Union[str, List[str]], opional)`: Defines the strategy for partitioning and recomputing Transformer blocks. Recompute method can be 'uniform' and 'block'.
- `recompute_num_layers (Union[int, List[int]], optional)`: Specifies the number of Transformer blocks to recompute per partition or stage. Interpretation depends on `recompute_method`.
- `recompute_layer_idxs_list (Union[int, List[int]], optional)`: Manually specifies the indices of Transformer blocks to recompute. It will override `recompute_method` and `recompute_num_layers` when set.

```python
def generate_gpt_hetero_4d_config(cp_list, rank_to_device_mapping, unused_rank, hetero_layers, accumulate_hetero_stages, num_layers, num_gpus, dp, tp, zero, recompute_granularity, recompute_method, recompute_num_layers, recompute_layer_idxs_list) -> List[Dict]:
```

Generates a ds_config file for heterogeneous distributed training based on the model and distribution configurations.

**Parameters**:

- `cp_list (List)`: Specifies the context parallelism (CP) configuration within each data parallel (DP) group.
  - For example, cp_list=[1,3] means that DP is set to 2:
    - The first pipeline forms the first DP group with CP=1.
    - The second, third, and fourth pipelines form the second DP group with CP=3.
- `rank_to_device_mapping (dict)`: A mapping between rank IDs and device IDs.
- `unused_rank (List)`: A list of rank IDs corresponding to faulty or unused GPUs.
- `hetero_layers (List[List])`: Specifies the distribution of model layers across different pipelines.
  - Example: If cp_list=[1,3], then hetero_layers=[[16, 16], [16, 16], [32], [32]] indicates:
    - Pipelines 1 and 2 are split into two stages, each containing 16 layers.
    - Pipelines 3 and 4 form a single stage with 32 layers each.
- `accumulate_hetero_stages (List)`: Stores the prefix sum of the number of stages in each pipeline.
  - Example: Given hetero_layers=[[16, 16], [16, 16], [32], [32]], the corresponding accumulate_hetero_stages would be [0,2,4,5,6].
- `num_layers (int, default=32)`: The total number of layers in the model.
- `num_gpus (int, default=8)`: The number of GPUs used for training.
- `dp (int, default=2)`: The degree of data parallelism (DP).
- `tp (int, default=2)`: The degree of tensor parallelism (TP).
- `zero (bool, default=True)`: Whether to enable ZeRO-1 optimization.
- `recompute_granularity (Union[str, List[str]], optional)`: The granularity of recomputation that controls the scope of activations to recompute within a Transformer block. It can be a string or a list of strings for heterogeneous recomputation. Recompute granularity can be 'selective' and 'full'.
- `recompute_method (Union[str, List[str]], opional)`: Defines the strategy for partitioning and recomputing Transformer blocks. Recompute method can be 'uniform' and 'block'.
- `recompute_num_layers (Union[int, List[int]], optional)`: Specifies the number of Transformer blocks to recompute per partition or stage. Interpretation depends on `recompute_method`.
- `recompute_layer_idxs_list (Union[int, List[int]], optional)`: Manually specifies the indices of Transformer blocks to recompute. It will override `recompute_method` and `recompute_num_layers` when set.

## Llama

### LlamaConfig

```python
class LlamaConfig(
  vocab_size=50257,
  hidden_size=4096,
  intermediate_size=11008,
  num_hidden_layers=32,
  num_attention_heads=32,
  num_key_value_heads=None,
  hidden_act="fast-swiglu",
  max_position_embeddings=2048,
  initializer_range=0.02,
  rms_norm_eps=1e-6,
  use_cache=True,
  pad_token_id=None,
  bos_token_id=1,
  eos_token_id=2,
  tie_word_embeddings=False,
  rope_theta=10000.0,
  rope_scaling=None,
  attention_bias=False,
  attention_dropout=0.0,
  mlp_bias=False,
  head_dim=None,
  use_flash_attn=True,
  gated_linear_unit=True,
)
```

LlamaConfig is the configuration class for Llama, defining various hyperparameters of the model.

**Parameters**:

- `vocab_size`: The size of the vocabulary.
- `hidden_size`: The dimension of the hidden representations.
- `intermediate_size`: Dimension of the MLP representations.
- `num_hidden_layers`: The number of layers in the Transformer network.
- `num_attention_heads`: The number of self-attention heads in each Transformer layer.
- `num_key_value_heads`: This is the number of key_value heads that should be used to implement Grouped Query Attention. If num_key_value_heads=num_attention_heads, the model will use Multi Head Attention (MHA), if num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used.
- `hidden_act`: The non-linear activation function (function or string) in the decoder.
- `max_position_embeddings`: The maximum sequence length that this model might ever be used with.
- `initializer_range`: The range of weight initialization, used to initialize the model parameters.
- `rms_norm_eps`: The epsilon value used in rms normalization to prevent division by zero errors.
- `use_cache`: Whether or not the model should return the last key/values attentions.
- `pad_token_id`: The ID of the pad_token (padding token), used to pad sequences to the same length.
- `bos_token_id`: The ID of the bos_token (begin-of-sequence token), representing the start of a sequence.
- `eos_token_id`: The ID of the eos_token (end-of-sequence token), representing the end of a sequence.
- `tie_word_embeddings`: Whether to tie weight embeddings.
- `rope_theta`: The base period of the RoPE embeddings.
- `rope_scaling`: The scaling factor for RoPE embeddings.
- `attention_bias`: Whether to use a bias in the query, key, value and output projection layers during self-attention.
- `attention_dropout`: The dropout ratio for the self-attention mechanism.
- `mlp_bias`: Whether to use a bias in the MLP layers.
- `head_dim`: The dimension of each attention head. If not specified, it is calculated as hidden_size / num_attention_heads.
- `use_flash_attn`: Whether to enable FlashAttention acceleration.
- `gated_linear_unit`: Whether to use a gated linear unit (GLU) in the MLP layers.

### LlamaModel

```python
LLaMALMHeadModel(config, ds_parallel_configs)->ht.nn.Module
```

Generates a LLaMA model based on the given model configuration and distributed system settings.

**Parameters**:

- `config (LlamaConfig)`: An instance of the LLaMAConfig class that defines the model's architecture and hyperparameters.
- `ds_parallel_configs (Dict)`: A list of distributed parallel configs.

### Llama ds config

```python
def generate_llama_4d_config(num_layers, num_gpus, dp, cp, tp, pp, zero recompute_granularity, recompute_method, recompute_num_layers, recompute_layer_idxs_list) -> List[Dict]
```

Generates a ds_config file based on the model architecture and parallelization strategy.

**Parameters**:

- `num_layers (int, default=32)`: The total number of layers in the model.
- `num_gpus (int, default=8)`: The number of GPUs used for training.
- `dp (int, default=2)`: The degree of data parallelism (DP).
- `cp (int, default=1)`: The degree of context parallelism (CP).
- `tp (int, default=2)`: The degree of tensor parallelism (TP).
- `pp (int, default=2)`: The degree of pipeline parallelism (PP).
- `zero (bool, default=True)`: Whether to enable ZeRO-1 optimization.
- `recompute_granularity (Union[str, List[str]], optional)`: The granularity of recomputation that controls the scope of activations to recompute within a Transformer block. It can be a string or a list of strings for heterogeneous recomputation. Recompute granularity can be 'selective' and 'full'.
- `recompute_method (Union[str, List[str]], opional)`: Defines the strategy for partitioning and recomputing Transformer blocks. Recompute method can be 'uniform' and 'block'.
- `recompute_num_layers (Union[int, List[int]], optional)`: Specifies the number of Transformer blocks to recompute per partition or stage. Interpretation depends on `recompute_method`.
- `recompute_layer_idxs_list (Union[int, List[int]], optional)`: Manually specifies the indices of Transformer blocks to recompute. It will override `recompute_method` and `recompute_num_layers` when set.

```python
def generate_llama_hetero_4d_config(cp_list, rank_to_device_mapping, unused_rank, hetero_layers, accumulate_hetero_stages, num_layers, num_gpus, dp, tp, zero, recompute_granularity, recompute_method, recompute_num_layers, recompute_layer_idxs_list) -> List[Dict]:
```

Generates a ds_config file for heterogeneous distributed training based on the model and distribution configurations.

**Parameters**:

- `cp_list (List)`: Specifies the context parallelism (CP) configuration within each data parallel (DP) group.
  - For example, cp_list=[1,3] means that DP is set to 2:
    - The first pipeline forms the first DP group with CP=1.
    - The second, third, and fourth pipelines form the second DP group with CP=3.
- `rank_to_device_mapping (dict)`: A mapping between rank IDs and device IDs.
- `unused_rank (List)`: A list of rank IDs corresponding to faulty or unused GPUs.
- `hetero_layers (List[List])`: Specifies the distribution of model layers across different pipelines.
  - Example: If cp_list=[1,3], then hetero_layers=[[16, 16], [16, 16], [32], [32]] indicates:
    - Pipelines 1 and 2 are split into two stages, each containing 16 layers.
    - Pipelines 3 and 4 form a single stage with 32 layers each.
- `accumulate_hetero_stages (List)`: Stores the prefix sum of the number of stages in each pipeline.
  - Example: Given hetero_layers=[[16, 16], [16, 16], [32], [32]], the corresponding accumulate_hetero_stages would be [0,2,4,5,6].
- `num_layers (int, default=32)`: The total number of layers in the model.
- `num_gpus (int, default=8)`: The number of GPUs used for training.
- `dp (int, default=2)`: The degree of data parallelism (DP).
- `tp (int, default=2)`: The degree of tensor parallelism (TP).
- `zero (bool, default=True)`: Whether to enable ZeRO-1 optimization.
- `recompute_granularity (Union[str, List[str]], optional)`: The granularity of recomputation that controls the scope of activations to recompute within a Transformer block. It can be a string or a list of strings for heterogeneous recomputation. Recompute granularity can be 'selective' and 'full'.
- `recompute_method (Union[str, List[str]], opional)`: Defines the strategy for partitioning and recomputing Transformer blocks. Recompute method can be 'uniform' and 'block'.
- `recompute_num_layers (Union[int, List[int]], optional)`: Specifies the number of Transformer blocks to recompute per partition or stage. Interpretation depends on `recompute_method`.
- `recompute_layer_idxs_list (Union[int, List[int]], optional)`: Manually specifies the indices of Transformer blocks to recompute. It will override `recompute_method` and `recompute_num_layers` when set.
