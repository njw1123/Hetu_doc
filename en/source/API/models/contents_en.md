
### Models

#### LLaMA

```python
class LLaMAConfig(
        vocab_size=50257,
        n_embd=768,
        ffn_hidden_size=-1,
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
        use_flash_attn = False,
        lora_dtype="fp32",
        dqtype="fp32",
 )
```

LLaMAConfig is the configuration class for LLaMA, defining various hyperparameters of the model.

**Parameters**:
- vocab_size: The size of the vocabulary.
- n_embd: The dimension of the embedding layer (i.e., the dimensionality of the representation vector for each token).
- ffn_hidden_size: The hidden layer size of the Feed-Forward Network (FFN). 
- n_layer: The number of layers in the Transformer network.
- n_head: The number of self-attention heads in each Transformer layer.
- n_inner: If set, it represents the dimensionality of the inner layer of the feed-forward network, typically larger than n_embd.
- activation_function: The type of activation function.
- resid_pdrop: The dropout ratio for residual connections.
- embd_pdrop: The dropout ratio for the embedding layer.
- attn_pdrop: The dropout ratio for the self-attention mechanism.
- layer_norm_epsilon: The epsilon value used in layer normalization to prevent division by zero errors.
- initializer_range: The range of weight initialization, used to initialize the model parameters.
- summary_type: The type of summary output during model training. Common values include cls_index (used for classification tasks), specifying how the model generates task-related summaries.
- summary_use_proj: Whether to use an additional linear transformation to project the model output into the label space. If set to True, an extra projection layer is applied.
- summary_activation: The activation function used for summary output, typically used in classification tasks.
- summary_proj_to_labels: Whether to project the summary output to the label space. If set to True, the model maps the final output layer to the label space.
- summary_first_dropout: The dropout ratio used when generating summaries.
- scale_attn_weights: Whether to scale attention weights.
- use_cache: Whether to enable caching.
- bos_token_id: The ID of the bos_token (begin-of-sequence token), representing the start of a sequence.
- eos_token_id: The ID of the eos_token (end-of-sequence token), representing the end of a sequence.
- scale_attn_by_inverse_layer_idx: Whether to scale attention weights inversely by layer index.
- reorder_and_upcast_attn: Whether to reorder and upcast precision during attention computation.
- use_flash_attn: Whether to enable FlashAttention acceleration.
- lora_dtype: The data type used for LoRA (Low-Rank Adaptation).
- dqtype: The data quantization type.

```python
LLaMALMHeadModel(config, ds_parallel_configs)->ht.nn.Module
```

Generates a LLaMA model based on the given model configuration and distributed system settings.

**Parameters**:
- config (LLaMAConfig): An instance of the LLaMAConfig class that defines the model's architecture and hyperparameters.
- ds_parallel_configs (Dict): A dictionary containing the distributed parallelism configurations.



```python
def generate_llama_4d_config(recompute_layers, num_layers=32, num_gpus=8, dp=2, cp=1, tp=2, pp=2, zero=True)->List[Dict]
```

Generates a ds_config file based on the model architecture and parallelization strategy.

**Parameters**:
- recompute_layers (List): Specifies the layers that should be recomputed during training.
- num_layers (int, default=32): The total number of layers in the model.
- num_gpus (int, default=8): The number of GPUs used for training.
- dp (int, default=2): The degree of data parallelism (DP).
- cp (int, default=1): The degree of context parallelism (CP).
- tp (int, default=2): The degree of tensor parallelism (TP).
- pp (int, default=2): The degree of pipeline parallelism (PP).
- zero (bool, default=True): Whether to enable ZeRO-1 optimization.



```python
def generate_llama_hetero_4d_config(cp_list, rank_to_device_mapping, unused_rank, hetero_layers, accumulate_hetero_stages, recompute_layers, num_layers=32, num_gpus=8, dp=2, tp=2, zero=True)->List[Dict]:
```
Generates a ds_config file for heterogeneous distributed training based on the model and distribution configurations.

**Parameters**:
- cp_list (List): Specifies the context parallelism (CP) configuration within each data parallel (DP) group.
  - For example, cp_list=[1,3] means that DP is set to 2:
    - The first pipeline forms the first DP group with CP=1.
    - The second, third, and fourth pipelines form the second DP group with CP=3.
- rank_to_device_mapping (dict): A mapping between rank IDs and device IDs.
- unused_rank (List): A list of rank IDs corresponding to faulty or unused GPUs.
- hetero_layers (List[List]): Specifies the distribution of model layers across different pipelines.
  - Example: If cp_list=[1,3], then hetero_layers=[[16, 16], [16, 16], [32], [32]] indicates:
    - Pipelines 1 and 2 are split into two stages, each containing 16 layers.
    - Pipelines 3 and 4 form a single stage with 32 layers each.
- accumulate_hetero_stages (List): Stores the prefix sum of the number of stages in each pipeline.
  - Example: Given hetero_layers=[[16, 16], [16, 16], [32], [32]], the corresponding accumulate_hetero_stages would be [0,2,4,5,6].
- recompute_layers (List): Specifies which layers should be recomputed during training.
- num_layers (int, default=32): The total number of layers in the model.
- num_gpus (int, default=8): The number of GPUs used for training.
- dp (int, default=2): The degree of data parallelism (DP).
- tp (int, default=2): The degree of tensor parallelism (TP).
- zero (bool, default=True): Whether to enable ZeRO-1 optimization.