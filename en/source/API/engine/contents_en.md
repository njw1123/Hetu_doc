# Training Engine

Hetu provides Trainer and SFTTrainer for pretraining and supervised fine-tuning tasks, respectively.

## Trainer

```python
class Trainer(pretrain_config, model, tokenizer, optimizer, train_dataset, data_collator, **kwargs)
```

Trainer class for training language models with distributed parallelism.

**Parameters:**

- `pretrain_config (TrainingConfig)`: Training configuration.
- `model (Union[PreTrainedModel, ModelWrapper, ModelWrapperFromConfig])`: Model to train.
- `tokenizer (Union[BaseTokenizer, DictConfig])`: Tokenizer for text processing.
- `optimizer (Union[OptimizerWrapper, DictConfig])`: Training optimizer.
- `train_dataset (Dataset, optional)`: Training dataset.
- `data_collator (Callable, optional)`: Custom data collator.
- `**kwargs`: Additional arguments.

### Training Config

```python
@dataclass
class TrainingConfig(
    output_dir,
    overwrite_output_dir,
    plot_loss,
    plot_update_freq,
    bf16,
    packing,
    micro_batch_size,
    ds_parallel,
    global_load_size,
    data_load_level,
    torch_profile,
    start_profile_step,
    end_profile_step,
    profile_save_path,
    train_dataset_path,
    dataset_text_field,
    max_seq_length,
    steps,
    learning_rate,
)
```

**Parameters:**

- `output_dir (str)`: Directory to save the training outputs.
- `overwrite_output_dir (bool)`: Whether to overwrite the output directory.
- `plot_loss (bool)`: Whether to plot the loss.
- `plot_update_freq (int)`: Frequency to plot the loss.
- `bf16 (bool)`: Whether to use bfloat16.
- `packing (bool)`: Whether to pack input sequences for faster training.
- `micro_batch_size (int, optional)`: Micro batch size for training (padding mode).
- `ds_parallel (StrategyConfig, optional)`: Data parallel configuration.
- `global_load_size (int)`: Global load size for data loading (samples/tokens).
- `data_load_level (DataLoadLevel)`: Data loading level (sample/token).
- `torch_profile (bool)`: Whether to profile PyTorch operations.
- `start_profile_step (int)`: Step to start profiling.
- `end_profile_step (int)`: Step to end profiling.
- `profile_save_path (str)`: Path to save profiling results.
- `train_dataset_path (str, optional)`: Path to training dataset.
- `dataset_text_field (str, optional)`: Dataset text field name.
- `max_seq_length (int, optional)`: Maximum sequence length.
- `steps (int)`: Number of training steps.
- `learning_rate (float)`: Learning rate.

**Methods:**

### get_train_data_loader()

```python
def get_train_data_loader() -> DataLoader
```

Get a data loader for training.

**Returns:**

- `DataLoader`: Configured data loader for training.

**Raises:**

- `ValueError`: If required dataset paths are not provided.

### build()

```python
def build()
```

Initialize training graph and optimizer.

**Raises:**

- `ValueError`: If trainer has already been built.

### train()

```python
def train(cp_list, strategy_id, run_level)
```

Main training loop with distributed execution

**Parameters:**

- `cp_list (List[int], optional)`: Optional list of communication parallelism degrees. If None, uses the default from ds_config.
- `strategy_id (int, default=0)`: ID of the parallel strategy to use.
- `run_level (hetu.run_level, default=hetu.run_level("update"))`: The execution level (update, grad, or forward).

### save_model()

```python
def save_model(output_dir=None, save_dtype=hetu.float32)
```

Save trained model weights

**Parameters:**

- `output_dir (str)`: Output directory.
- `save_dtype`: Data type for saving weights.

**Examples:**

```python
# Basic usage
trainer = Trainer(
    pretrain_config=config,
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
)
trainer.build()
trainer.train()
trainer.save_model("output_dir")
```

## SFTTrainer

SFTTrainer is a specialized trainer for supervised fine-tuning tasks. It inherits from the Trainer class and adds additional functionality specific to supervised fine-tuning.

```python
class SFTTrainer(sft_config, model, tokenizer, optimizer, train_dataset, data_collator, message_template, prompt_template, **kwargs)
```

**Parameters:**

- `sft_config (SFTConfig)`: Configuration for supervised fine-tuning.
- `model (Union[PreTrainedModel, ModelWrapper, ModelWrapperFromConfig])`: Model to train.
- `tokenizer (Union[BaseTokenizer, DictConfig])`: Tokenizer for text processing.
- `optimizer (Union[OptimizerWrapper, DictConfig])`: Training optimizer.
- `train_dataset (Dataset, optional)`: Training dataset.
- `data_collator (Callable, optional)`: Custom data collator.
- `message_template (str)`: Template for messages.
- `prompt_template (str)`: Template for prompts.
- `**kwargs`: Additional arguments.

### SFT Config

```python
class SFTConfig(peft, prompt_template)
```

`SFTConfig` inherits from `TrainingConfig` and is used to configure the supervised fine-tuning process. It includes parameters for PEFT (Parameter-Efficient Fine-Tuning) and prompt templates.

**Parameters:**

- `peft (Dict)`: PEFT configuration such as LoRA config.
- `prompt_template (str)`: Template for prompts.
