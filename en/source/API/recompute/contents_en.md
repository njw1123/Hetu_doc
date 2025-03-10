### Recompute

Hetu Supports Activating Recomputing for Specific Layers Within a Designated Data Parallel Group

**Usage:**
To enable recomputation for specific layers in the model, modify the recompute_layers parameter in train_hetu.sh. This parameter specifies which layers should undergo recomputation.

During training, train_hetu.sh passes this parameter to generate_llama_4d_config.py, which then calls generate_gpt_4d_config and utilizes recompute_layers to generate the corresponding configuration.

