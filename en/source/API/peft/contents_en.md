# Peft

## LoRA

LoRA (Low-Rank Adaptation) is a technique for fine-tuning large language models (LLMs) with a small number of trainable parameters. It works by adding low-rank matrices to the original weight matrices of the model, allowing for efficient adaptation without modifying the original weights. This approach significantly reduces the computational cost and memory requirements for training, making it feasible to fine-tune large models on smaller datasets or with limited resources.

Hetu supports adding LoRA to pretrained models, enabling users to adapt these models to specific tasks or domains with minimal effort. Moreover, Hetu also supports multi-task LoRA fine-tuning.

## LoRA Config

```python
class LoraConfig(rank, target_modules, lora_alpha, lora_dropout, use_rslora)
```

**Parameters:**

- `rank (int)`: The rank of the low-rank matrices.
- `target_modules (List[str])`: The names of the modules to which LoRA will be applied.
- `lora_alpha (float)`: The scaling factor for the LoRA matrices.
- `lora_dropout (float)`: The dropout rate for the LoRA matrices.
- `use_rslora (bool)`: Whether to use Rank-Stabilized LoRA.

## Example: How to build a LoRA model

```python
# Load pretrained model
pretrained_model = LlamaLMHeadModel.from_pretrained(
    pretrained_model_name_or_path,
    ds_parallel_configs,
)

# Build LoRA model
model = LoraModel(
    model=pretrained_model,
    peft_configs=lora_configs,
    config=pretrained_model.model_config
)

# Build multi-task LoRA model
model = MultiLoraModel(
    model=pretrained_model,
    peft_configs=lora_configs,
    config=pretrained_model.model_config
)
```
