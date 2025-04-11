# Data Loading and Processing

## Dataset

### JsonDataset

```python
class JsonDataset(json_file, text_field, tokenizer, max_seq_len)
```

Dataset class for loading and processing JSON format text data. This class inherits from torch.utils.data.Dataset.

**Parameters:**

- `json_file (str)`: Path to the JSON file containing the dataset.
- `text_field (str)`: JSON field name specifying where the data is stored.
- `tokenizer (BaseTokenizer)`: Tokenizer for encoding text data.
- `max_seq_len (int)`: The maximum length of input sequences.

**Functions:**

- `pad_id()`: Returns the pad token id.
- `len()`: Returns the size of the dataset.
- `getitem(idx)`: Returns a dict of the idx-th data sample, including `input_ids`.

**Examples:**

```python
# load 'content' field from root_folder/test.json as dataset
root_folder = "data"
test_dataset = JsonDataset(
    json_file=os.path.join(root_folder, "test.json"),
    text_field="content",
    tokenizer=tokenizer,
    max_seq_len=1024
)
```

### SFTDataset

```python
class SFTDataset(json_file, tokenizer, max_seq_len, message_template, prompt_template)
```

Dataset class for supervised fine-tuning with structured message format. This dataset handles loading and processing data for supervised fine-tuning, including message formatting and tokenization.

**Parameters:**

- `json_file (str)`: Path to the JSON file containing the dataset.
- `tokenizer (PreTrainedTokenizer)`: Tokenizer for encoding text data.
- `max_seq_len (int)`: The maximum length of input sequences.
- `message_template (MessageTemplate)`: Template for formatting messages
- `prompt_template (str, optional)`: Template for formatting prompts

**Functions:**

- `pad_id()`: Returns the pad token id.
- `len()`: Returns the size of the dataset.
- `getitem(idx)`: Returns a dict of the idx-th data sample, including `input_ids` and `label_mask`.

## DataLoader

### build_data_loader

```python
build_data_loader(dataset, consumed_samples, global_batch_size, data_load_level, data_collator)
```

Build a DataLoader with specified batch sampling strategy. This function creates a DataLoader that supports two loading levels:
    - Sample level: batch size is determined by number of samples
    - Token level: batch size is determined by total number of tokens

**Parameters:**

- `dataset (torch.utils.data.Dataset)`: The dataset to load.
- `consumed_samples (int)`: Number of samples already consumed.
- `global_load_size (int, optional)`: Global batch size (samples) or token number (tokens). Defaults to None.
- `data_load_level (DataLoadLevel, optional)`: Loading strategy, either "sample" or "token". Defaults to "sample".
- `data_collator (Callable, optional)`: Custom collator function. Defaults to None.

**Returns:**

- `torch.utils.data.DataLoader`: The configured data loader.

**Examples:**

```python
loader = build_data_loader(
    dataset=my_dataset,
    consumed_samples=0,
    global_load_size=2048,
    data_load_level="token"
)
```

### build_dist_data_loader

```python
build_dist_data_loader(dataset, consumed_samples, micro_batch_size, dp_rank, dp_size, data_load_level)
```

Build a distributed DataLoader for parallel training. This function creates a DataLoader suitable for distributed training scenarios. It currently only supports sample-level batch sampling.

**Parameters:**

- `dataset (torch.utils.data.Dataset)`: The dataset to load.
- `consumed_samples (int)`: Number of samples already consumed.
- `micro_batch_size (int)`: Size of micro batches for each data parallel worker.
- `dp_rank (int)`: Rank of current data parallel process.
- `dp_size (int)`: Total number of data parallel processes.
- `data_load_level (DATA_LOAD_LEVEL, optional)`: Loading strategy, must be "sample". Defaults to "sample".

**Returns:**

- `torch.utils.data.DataLoader`: The configured distributed data loader.

**Examples:**

```python
loader = build_dist_data_loader(
    dataset=my_dataset,
    consumed_samples=0,
    micro_batch_size=32,
    dp_rank=0,
    dp_size=8,
    data_load_level="sample"
)
```

**Raises:**

AssertionError: If data_load_level is not "sample".

## Bucket

```python
class Bucket(pad_token, max_seqlen, alignment)
```

Bucket is a container for a batch of data, which is used for padding and packing. The inner batch is a dictionary of numpy arrays, where the key is the field name such as 'input_ids' and 'labels'.

**Parameters:**

- `pad_id (int)`: The padding token id.
- `max_seqlen (int)`: The maximum sequence length.
- `alignment (int)`: Ensures sequence lengths are aligned to a multiple of this value, used for packing.

**Methods:**

- `add_data(data, truncate_length)`: Adds a sequence to the bucket, where truncate_length indicates the truncation length. If truncate_length is None, the sequence is added as is. Pad tokens in data will be ignored.
- `pad_data()`: Pads all sequences in the bucket to max_seqlen.
- `pack_data(batching_option_matrix, static_shape, sorted = True)`: Pack data in the bucket to micro batches. There are two packing strategies: workload balance and greedy. The workload balance strategy is controlled by the batching_option_matrix. The greedy strategy is controlled by the static_shape and sorted.
  - `batching_option_matrix (numpy.ndarray, optional)`: Workload balance seq-batch mapping. The element at row i, column j indicates whether the i-th sequence belongs to the j-th micro-batch. It can be a 2D NumPy ndarray or None. If not provided, a greedy packing strategy is applied.
  - `static_shape (bool, optional)`: Indicating whether to use static shape (max sequence length) for all micro batches. If True, the batch size is fixed and the sequences are packed into micro batches of the same size, filled by pad tokens if necessary. If False, the batch size can vary.
  - `sorted (bool)`: Whether the sequences are sorted by length before packing. If False, the original order is preserved.
- `generate_cp_pack_data(cp_size, cp_lens_rate)`: Generate packed data for context parallel.
  - `cp_size (int)`: The number of context parallel ranks.
  - `cp_lens_rate (List[float], optional)`: The ratio of sequence lengths for each context parallel rank.
- `packed_batch_size()`: Returns the size of packed batch.
- `padded_batch_size()`: Returns the size of padded batch.
- `original_batch_size()`: Returns the size of original batch before padding and packing.
- `padded_batch()`: Returns a list of padded sequences, each as a 1D NumPy array.
- `packed_batch()`: Returns a list of packed sequences, each as a 1D NumPy array.
- `packed_cu_seqlens_list()`: Returns a list of cumulative sequence lengths after packing. For example, [0, 1024, 2048] indicates two sequences of length 1024 concatenated into one.
- `cp_packed_batch (int)`: Return the packed batch of the specified context parallel rank.
- `cp_packed_cu_seqlens_list()`: Returns a list of cumulative sequence lengths of context parallel packed batch.
- `cp_packed_seqlen_list()`: Return a dict of packed sequence length list for each context parallel rank.

### get_sorted_batch_and_len

```python
get_sorted_batch_and_len(global_batch, pad_token) -> (numpy.ndarray, numpy.ndarray)
```

Sort the global batch by the number of non-padding tokens in ascending order.

**Parameters:**

- `global_batch (Union[numpy.ndarray, Dict[str, numpy.ndarray]])`: Global batch of 2D NumPy array where each row represents a sequence.
- `pad_token (int)`: The pad token id of the tokenizer.

**Returns:**

- `sorted_global_batch (numpy.ndarray)`: The batch sorted by the number of non-pad tokens.
- `sorted_valid_tokens (numpy.ndarray)`: The count of valid (non-pad) tokens for each sequence.

### get_input_and_label_buckets

```python
get_input_and_label_buckets(global_batch, pad_id, batch_indices, max_seqlen, alignment, valid_alignment) -> Bucket
```

Selects data for the current data parallel group based on batch_indices, and creates a Bucket containing `input_ids` and `labels` fields.

**Parameters:**

- `global_batch (Union[numpy.ndarray, Dict[str, numpy.ndarray]])`: Global batch of 2D NumPy array where each row represents a sequence.
- `pad_id (Union[int, Dict[str, int]])`: Pad token id of the tokenizer. If a dict is provided, it should contain the pad token id for each field in the global batch.
- `batch_indices (List[int])`: Indices to slice the global_batch and select relevant data.
- `max_seqlen (int)`: Maximum sequence length.
- `alignment (int, optional)`: Ensures sequence lengths are aligned to a multiple of this value. Defaults to 128.
- `valid_alignment (int, optional)`: Ensures valid sequence lengths are aligned to a multiple of this value in packing. Defaults to 1.

**Returns:**

- A bucket containing both `input_ids` and `labels` fields.

## Tokenizer

Hetu provides various tokenizer types, including GPT2, HuggingFace, SentencePiece and TikToken. The implementation of each tokenizer type are encapsulated in the `BaseTokenizer` class, which is the base class for all tokenizers. The `BaseTokenizer` class provides common methods for tokenization, encoding, and decoding. The details can be found in `hetu.data.tokenizers`. Here we will introduce the common usage in pretraining and supervised fine-tuning.

### build_tokenizer

Pretraining will use `build_tokenizer` to build a tokenizer. The function will build the tokenizer type based on the provided parameters. You are required to provide args in `kwargs` for specific tokenizer types, such as `vocab_file` and `merge_file` for `GPT2BPETokenizer`.

```python
build_tokenizer(tokenizer_type, rank, make_vocab_size_divisible_by, tensor_model_parallel_size, vocab_extra_ids, **kwargs)
```

### PreTrainedTokenizer

For supervised fine-tuning, the tokenizer is often provided by pretrained models. `PreTrainedTokenizer` is a wrapper for different tokenizer types and you can access the backend tokenizer through `tokenizer_class` and `backend_tokenizer` properties.

**Parameters:**

- `tokenizer_type`: Type of tokenizer to build (e.g., 'GPT2BPETokenizer', 'SentencePieceTokenizer').
- `rank`: Process rank.
- `make_vocab_size_divisible_by`: Make the vocabulary size divisible by this value.
- `tensor_model_parallel_size`: Tensor model parallel size.
- `vocab_extra_ids`: Number of extra IDs to add to the vocabulary.
- `**kwargs`: Additional keyword arguments for the tokenizer.

**Class Methods:**

```python
from_pretrained(pretrained_model_name_or_path, cache_dir, **kwargs) -> PreTrainedTokenizer
```

Load a tokenizer from a pre-trained model.

**Parameters:**

- `pretrained_model_name_or_path`: Directory path or name of the model.
- `cache_dir`: Directory where model files will be cached.
- `kwargs`: Additional keyword arguments.
  - `subfolder`: Optional subfolder within the model directory.
  - `pattern`: Regex pattern for tiktoken.
  - `special_tokens`: Special tokens for tiktoken.
  - `tokenizer_class`: The tokenizer class name.

## Template

Hetu provides message and prompt templates for supervised fine-tuning. The templates are used to format the input data into a structured format suitable for training.

### Message Template

Message template is used to format the input data into a list of messages. A message contains 'role', 'content' and 'masked' fileds, where 'masked' indicates whether the content is masked or not.

All message templates inherit from protocol `MessageTemplate` and thus a message should implement its `__call__` function to convet a dataset sample to a list of messages.

```python
def __call__(self, sample) -> List[Mapping[str, Any]]
```

**Parameters:**

- `sample (Mapping[str, Any])`: A sample from the dataset.

**Returns:**

- A list of message dictionaries, each with 'role', 'content', and 'masked' keys.

Hetu provides some common message templates including `InputOutputTemplate`, `AlpacaTemplate`, `ShareGPTTemplate` and `OpenAITemplate`. Here we take `OpenAITemplate` as an example.

```python
class OpenAITemplate(train_on_input, column_map, new_system_prompt)
```

**Parameters:**

- train_on_input (bool, optional): Whether to train on the input messages. Defaults to False.
- column_map (Dict[str, str], optional): Mapping from standard column names to dataset column names. Defaults to None, which means {'messages': 'messages'}.
- new_system_prompt (str, optional): System prompt to add at the beginning. Defaults to None.

**Raises:**

ValueError: If the column_map doesn't contain a 'messages' key.

### Prompt Template

Prompt templates are structured text templates used to format a list of messages into a single sequence. Templates are strings of Jinja2 format with placeholders for dynamic content. The placeholders are replaced with actual values when the template is rendered.

Hetu provides some common prompt templates including `CHATML_TEMPLATE`, `SUMMARIZE_TEMPLATE` and `QWESTION_ANSWER_TEMPLATE`. Model-specific templates from HuggingFace models are also supported.

To use prompt template, you need to pass a Jinja2 template string to the `PromptTemplate` class and call it with a list of messages.

```python
def __call__(self, messages, return_mask, train_on_all_assistant, add_generation_prompt) -> str
```

**Parameters:**

- messages (List[Dict]): List of message dictionaries, each with 'role' and 'content' keys.
- return_mask (bool, optional): Whether to return position masks for training. Defaults to True.
- train_on_all_assistant (bool, optional): Whether to train on all assistant messages. Defaults to True.
- add_generation_prompt (bool, optional): Whether to add a generation prompt at the end. Defaults to False.

**Returns:**

- If return_mask=False: return the formatted message string.
- If return_mask=True: return a tuple of (formatted string, list of content position tuples). Each position tuple contains (start_position, end_position) for tracked message content, used for building label mask.

## DataCollator

### DataCollatorForLanguageModel

```python
class DataCollatorForLanguageModel()
```

Data collator used for language model training. It will generate 'labels' if 'label_mask' is provided in the batch, otherwise it will replace pad_id of 'input_ids' with -100 for 'labels'.

The returned batch will be a dictionary containing 'input_ids' and 'labels'.

## utils

### build_fake_batch_and_len

```python
build_fake_batch_and_len(fake_seqlens, pad_token, fake_fill_value) -> (numpy.ndarray, List[int])
```

Build a fake batch according to given seqlens of each sample using `fake_fill_value` as filler value and padding with `pad_token` to the maximum sequence length. The sequences inside the batch are sorted by lengths in ascending order. This function is useful for creating a batch of sequences for testing or debugging purposes.

**Parameters:**

- `fake_seqlens (List[int])`: List of sequence lengths for the fake sequences.
- `pad_token (int)`: The pad token id.
- `fake_fill_value (int, optional)`: The value to fill in the fake sequences. Defaults to 0.

**Returns:**

- `fake_batch (numpy.ndarray)`: A 2D NumPy array containing the padded fake sequences.
- `valid_lengths (List[int])`: A list of the original (unpadded) sequence lengths.

### get_mask_and_position_ids

```python
get_mask_and_position_ids(tokens, pad_id) -> (numpy.ndarray, numpy.ndarray)
```

Generate attention mask and position ids for input tokens.

**Parameters:**

- `tokens (numpy.ndarray)`: Input tokens of shape [batch_size, seq_length].
- `pad_id (int)`: Padding token id.

**Returns:**

- `attention_mask (numpy.ndarray)`: Attention mask of shape [batch_size, seq_length]
- `position_ids (numpy.ndarray)`: Position ids of shape [batch_size, seq_length]

### convert_parquet_to_json

```python
convert_parquet_to_json(parquet_file, json_file, columns, chunksize)
```

Convert parquet file to json file.

**Parameters:**

- `parquet_file (str)`: path to the parquet file.
- `json_file (str)`: path to the json file. Defaults to None.
- `columns (List[str], optional)`: list of columns to be converted. Defaults to None.
- `chunksize (int, optional)`: chunk size for reading the parquet file. Defaults to None.

**Returns:**

- A json file containing the converted data.
