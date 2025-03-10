#### hetu.embedding_lookup

```
hetu.embedding_lookup(input: hetu.tensor, id: hetu.tensor, multi_offset: List[int] = [0]) -> hetu.tensor
```

Looks up vectors from a partitioned embedding table based on given indices.

**Parameters:**

* input (hetu.tensor): The partitioned embedding matrix with shape (total_vocab_size, embed_dim).

* id (hetu.tensor): The index tensor, which can have any shape, with integer indices in the range [0, total_vocab_size).

* multi_offset (List[int], optional): A list of vocab start indices. In tensor parallelism, vocab embeddings are split within device groups, so each device needs to know its corresponding vocab start index.

**Returns:**

* hetu.tensor: The output tensor with shape id.shape + (embed_dim,).

