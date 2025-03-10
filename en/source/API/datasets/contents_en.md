### Datasets


#### HetuJsonDataset

```python
class HetuJsonDataset(json_file, key, max_seq_len, vocab_file, merge_file)
```

Reads a dataset in JSON format. This class inherits from torch's dataset.

**Parameters:**
- json_file (str): Path to the JSON file containing the dataset.
- key (str): A key in the JSON file specifying where the data is stored.
- max_seq_len (int): The maximum length of input sequences.
- vocab_file (str): Path to the vocabulary file.
- merge_file (str): Path to a merge file, typically used for token merging.

**Functions:**
- pad_id(): Returns the pad token id.
- len(): Returns the size of the dataset.
- getitem(idx): Returns the idx-th data.


#### build_data_loader

```python
build_data_loader(dataset, consumed_samples, global_batch_size=None, global_token_num=None)
```

Builds a data loader for a given dataset. It can either return batches with a fixed number of sequences (global_batch_size) or batches with a fixed number of tokens (global_token_num).

**Parameters:**
- dataset (torch.utils.data.Dataset): The dataset to be loaded.
- consumed_samples (int): The number of samples that have already been processed.
- global_batch_size (int, optional): The number of sequences in each global batch.
- global_token_num (int, optional): The total number of tokens in each global batch.

#### Bucket

```python
class Bucket(pad_token: int, max_seqlen: int, alignment: int)
```

A container for storing sequences, applying padding or packing strategies as needed.

**Parameters:**
- pad_token (int): The pad token ID.
- max_seqlen (int): The maximum sequence length.
- alignment (int): Ensures sequence lengths are aligned to a multiple of this value.

**Methods:**
- add_data(padded_sequence, valid_tokens): Adds a padded sequence to the bucket, where valid_tokens indicates the sequence length excluding padding.
- pad_data(): Pads all sequences in the bucket to max_seqlen.
- pack_data(batching_option_matrix, static_shape: bool, sorted=True): Packs sequences using a batching option matrix. If batching_option_matrix is a 2D NumPy array, the element at row i, column j indicates whether the i-th sequence belongs to the j-th micro-batch. If not provided, a greedy packing strategy is applied.
- packed_batch_size(): Returns the number of sequences after packing.
- padded_batch_size(): Returns the number of sequences after padding.
- original_batch_size(): Returns the number of sequences before padding and packing.
- padded_batch(): Returns a list of padded sequences, each as a 1D NumPy array.
- packed_batch(): Returns a list of packed sequences, each as a 1D NumPy array.
- packed_cu_seqlens_list(): Returns a list of cumulative sequence lengths after packing. For example, [0, 1024, 2048] indicates two sequences of length 1024 concatenated into one.

#### get_sorted_batch_and_len

```python
get_sorted_batch_and_len(global_batch: np.ndarray, pad_token: int)->(np.ndarray, np.ndarray)
```
Sorts a global batch of sequences based on the number of non-padding tokens.

**Parameters:**
- global_batch (np.ndarray): A 2D NumPy array where each row represents a sequence.
- pad_token (int): The pad token ID.

**Returns:**
- sorted_global_batch (np.ndarray): The batch sorted by the number of non-pad tokens.
- valid_token_counts (np.ndarray): The count of valid (non-pad) tokens for each sequence.

#### build_fake_batch_and_len

```python
build_fake_batch_and_len(fake_seqlens: List[int], pad_token: int)->(np.ndaraay, List[int])
```
Constructs a fake batch of sequences using 0 as placeholder values and fills them with pad_token up to the maximum sequence length.

**Parameters:**
- fake_seqlens (List[int]): List of sequence lengths for the fake sequences.
- pad_token (int): The pad token ID.

**Returns:**
- fake_batch (np.ndarray): A 2D NumPy array containing the padded fake sequences.
- valid_lengths (List[int]): A list of the original (unpadded) sequence lengths.

#### get_input_and_label_buckets

```python
get_input_and_label_buckets(global_batch: np.ndarray, pad_token: int, batch_indices: List[int], max_seqlen: int, alignment: int)->(Bucket, Bucket)
```
Selects data for the current data parallel group based on batch_indices, and creates input_bucket and label_bucket.

**Parameters:**
- global_batch (np.ndarray): A 2D NumPy array where each row represents a sequence.
- pad_token (int): The pad token ID.
- batch_indices (List[int]): Indices to slice the global_batch and select relevant data.
- max_seqlen (int): Maximum sequence length.
- alignment (int): Ensures sequence lengths are aligned to a multiple of this value.

**Returns:**
- input_bucket (Bucket): A bucket containing the input sequences.
- label_bucket (Bucket): A bucket containing the corresponding label sequences.