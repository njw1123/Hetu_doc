### Checkpoint

#### save_checkpoint

```python
save_checkpoint(model, optimizer, path, config=None, local_device=None)->None
```
This function aggregates partitioned parameters and saves them.
In pipeline parallelism, parameters from different stages are stored in separate files, resulting in pp_num_stage checkpoint files.

**Parameters:**

- model (Module): The model defined using Hetu’s Module.
- optimizer (Optimizer): The optimizer defined using Hetu’s Optimizer.
- path (str): The directory path where the checkpoint files will be saved.
- config (Dict, optional): Model configuration, such as LLaMAConfig.
- local_device (device, optional): The device used by the current process.



#### load_checkpoint



```python
load_checkpoint(model, optim, file, config=None, local_device=None)->None
```

This function loads a previously saved checkpoint and restores the model and optimizer states.


**Parameters:**
- model (Module): The model defined using Hetu's Module.
- optim (Optimizer): The optimizer defined using Hetu's Optimizer.
- file (str): Path to the checkpoint file.
- config (Dict, optional): Configuration dictionary for the model (e.g., LLaMAConfig).
- local_device (device, optional): The device used by the current process.