# Utils

## distributed_init

```python
distributed_init(ngpus, server_addr, server_port, need_kv_store)
```

Initializes distributed communication and device groups. Automatically sets HETU_LOCAL_HOSTNAME environment variable if not present.

**Parameters:**

- `ngpus (int)`: Number of GPUs to use
- `server_addr (str, default="127.0.0.1")`: Communication server address
- `server_port (str, default="23457")`: Communication server port
- `need_kv_store (bool, default=False)`: Whether to initialize key-value store client

**Returns:**

- Tuple containing local device and device group (plus KV store client if needed)

## read_ds_parallel_config

```python
read_ds_parallel_config(ds_parallel_config, num_strategy)->List[Dict]
```

Reads the DistConfig file and adds zero configuration(whether to use zero1) for variable-type parameters.

**Parameters:**

- `ds_parallel_config (List[Dict])`: Path(s) to the DistConfig file. Multiple paths should be separated by commas (",").
- `num_strategy (int)`: Number of strategies.

## get_multi_ds_parallel_config

```python
get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range) -> List[Dict]
```

Extracts the distributed configuration for a specific module_name from DistConfig. This function only retrieves relevant information from DistConfig and returns it as a List[Dict]. The returned data is not the internal distributed state used by Hetu.

**Parameters:**

- `ds_parallel_configs (List[Dict])`: Distributed configuration obtained from read_ds_parallel_config.
- `module_name (str)`: Name of the module.
- `_range (int, default=-1)`: If the module appears in a specific layer, this parameter specifies the corresponding layer. Defaults to -1 (i.e., no specific layer).

## parse_multi_ds_parallel_config

```python
parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
```

Extracts the corresponding distributed configuration from DistConfig and converts it into the internal distributed state and device setup used by Hetu.

**Parameters:**

- `ds_parallel_configs (List[Dict])`: Distributed configuration obtained from read_ds_parallel_config.
- `module_name (str)`: Name of the module.
- `_range (int, default=-1)`: If the module appears in a specific layer, this parameter specifies the corresponding layer. Defaults to -1.

## config2ds

```python
config2ds(config)
```

Converts a distributed configuration into the internal distributed state and device setup used by Hetu.

**Parameters:**

- `config (dict)`: The distributed configuration.
