### DistributedStates

#### hetu.DistributedStates

```python
class hetu.DistributedStates(device_num:int, states:Dict[int, int], order:List[int]=None, zero:bool=False)
```

This class is used for managing distributed states across multiple devices in a distributed training setup. 

**Parameters**

- device_num(int): The number of devices participating in the distributed setup. 
- states(Dict[int, int]): A dictionary that describes the distribution of the data across the devices. The keys represent the state types, and the values represent how the data is distributed among the devices. For example:
  - -2: Represents partial distribution (e.g., the gradient before aggregation in data parallelism).
  - -1: Represents duplicate distribution
  - 0-(n-1): Represents data sliced along a specific dimension 
  - For instance, { -2: 2, -1: 2, 0: 2 } means the data is: Split into 2 partial parts, Duplicated into 2 parts, Split along the 0th dimension into 2 parts, All of which are distributed over 8 devices (e.g., GPUs).
- order (List[int], optional): Storing the keys from the states in an ordered manner, in order to map states to devices.”
- zero (bool, optional): Indicates whether zero1 is enabled.

**Properties**

- device_num(int): The number of devices participating in the distributed setup. 
- states(Dict[int, int]): A dictionary that describes the distribution of the data across the devices. 
- order (List[int]): Storing the keys from the states in an ordered manner. 
- zero (bool): Indicates whether zero1 is enabled.
- is_pure_duplicate(bool): A boolean that indicates whether each device stores the complete data without partitioning.

**Methods**

- check_equal(hetu.DistributedStates)->bool: Compares the current DistributedStates object with another to check if their state distributions are the same.
- get_dim(dim:int)->int: Retrieves the size of the specified dimension in the state distribution.
- get_dup_group_index(device_index:int)->int: Treats devices that store a complete copy of the parameters as a group, and queries which group the given device belongs to.
  
#### hetu.map_to_local_data

```python
hetu.map_to_local_data(ds:hetu.DistributedStates, device_index:int)->Dict[int, int]
```

**Parameters**

This function maps the device index to a local state in a distributed setup. It returns the local state identifier for the device in a distributed system. For example, given a distributed state configuration such as { -2: 2, -1: 2 }, the function returns possible mappings like { -2: 0, -1: 0 }, { -2: 0, -1: 1 }, { -2: 1, -1: 0 }, { -2: 1, -1: 1 }

- ds(hetu.DistributedStates): The DistributedStates object that contains the state distribution across devices. 
- device_index(index): The index of the device whose local state mapping is to be retrieved. 

#### hetu.DistributedStatesUnion

```python
class hetu.DistributedStatesUnion(ds_list:List[hetu.DistributedStates], hetero_dim:int)
```

The DistributedStatesUnion class is designed to manage and combine distributed states across multiple heterogeneous pipelines in a distributed system. Each pipeline may have different state distributions, and this class consolidates the states for all pipelines into a single union. It allows for efficient handling of distributed states in heterogeneous pipeline configurations, where the data splitting and distribution may differ between pipelines.

**Parameters**

- ds_list(List[hetu.DistributedStates]): A list where each element is a hetu.DistributedStates object representing the state distribution for each pipeline. 
- hetero_dim(int): The dimension along which data is split between different pipelines. 

**Properties**

- ds_list(List[hetu.DistributedStates]):A list of hetu.DistributedStates objects, each representing the state configuration for a different pipeline. 
- hetero_dim(int): The dimension along which the data is split between the different pipelines. 

**Methods**

- get(dim: int): Retrieves the distributed state of the specified pipeline at the given dimension.
- get_local(dim: int): Retrieves and converts the distributed state of the specified pipeline into its local state.  If there are two pipelines with distributed states `{ -1: 2, 0: 2 }` and `{ 0: 4 }`, and `hetero_dim=0`, their converted local states would be `{ -1: 2, 0: 1 }` and `{ 0: 2 }` respectively.
