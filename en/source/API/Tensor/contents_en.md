### Tensor

The Tensor class in Hetu represents a multi-dimensional array with associated properties and methods. 

**Properties**

- id(int): The unique identifier for the tensor.
- name(str): The name of the tensor. 
- ndim(int): The number of dimensions (axes) in the tensor.
- shape(List[int]): A list representing the shape of the tensor, where each element in the list corresponds to the size of the tensor along a particular axis. 
- device(hetu.device): The device on which the tensor is stored
- dtype(hetu.dtype): The data type of the tensor’s elements
- global_shape(List[int]): The global shape of the tensor, typically used in distributed systems where the tensor is split across multiple devices. 
- symbolic_shape(List[hetu.IntSymbol]): A list of symbolic shapes, where each element is an IntSymbol object.
- is_variable(bool): A boolean flag that indicates whether the tensor is a variable. 
- is_parameter(bool): A boolean flag that indicates whether the tensor is a model parameter.
- requires_grad(bool): A boolean flag that specifies whether the tensor requires gradients.
- distributed_states(hetu.DistributedStates): The DistributedStates object associated with the tensor. 
- ds_hierarchy(List[hetu.DistributedStatesUnion]): A list of DistributedStatesUnion objects representing the hierarchical distribution of the tensor’s states across multiple devices and pipelines.
- has_placement_group(bool): A boolean flag that indicates whether the tensor has a placement group.
- placement_group_union(List[hetu.DeviceGroup]): A list of hetu.DeviceGroup objects, where each DeviceGroup corresponds to a group of devices in a specific pipeline. 
- data(hetu.NDArray): The NDArray object that contains the actual data stored in the tensor. 
- graph(hetu.Graph): The Graph object that represents the computation graph associated with the tensor. 

**Methods**

- dim()->int: Returns the number of dimensions of the tensor.
- size(dim:int=None)->int: Returns the size of the tensor along the specified dimension.
- numel()->int: Returns the total number of elements in the tensor by multiplying the sizes of all dimensions.
- global_numel()->int: Returns the total number of elements in the tensor considering its global shape
- timecost()->int: Returns the time cost of the operation producing the tensor, measured in microseconds.
- stride(dim:int=None)->int: Returns the stride of the tensor along the specified dimension.
- numpy(force:bool=false, save:bool=false): Converts the tensor to a numpy array. 
  - force (bool): If True, and the NDArray is not on the CPU, the method will force the transfer of the data to the CPU before converting it into a numpy ndarray.
  - save (bool): If False, for float16 and bfloat16 data types, the original data is preserved as int16_t. If True, the data is converted to float32 for these types to ensure compatibility with numpy arrays
- to(dtype:hetu.dtype, dev:hetu.device=None, stream_index:int=None, device_group_hierarchy:List[List[hetu.DeviceGroup]] = None,  extra_deps:List[hetu.Tensor]=None, name:str=None, is_cpu:bool=None)->hetu.Tensor: 
  - dtype (hetu.dtype): The desired data type for the tensor.
  - dev (hetu.device, optional): The target device for the tensor (e.g., CPU, GPU). If not specified, the tensor remains on its current device.
  - stream_index (int, optional): The index of the stream for operation scheduling. If not specified, it defaults to None.
  - device_group_hierarchy (List[List[hetu.DeviceGroup]], optional): The device group hierarchy for distributed settings.
  - extra_deps (List[hetu.Tensor], optional): Additional tensors that might be required as dependencies during the conversion.
  - name (str, optional): The name of the tensor after the conversion.
  - is_cpu (bool, optional): If True, forces the tensor to be moved to the CPU.
- reset_data(provided_data:numpy.ndarray)->None: Resets the data in the tensor using a numpy array. 
- reset_data(provided_data:hetu.NDArray)->None: Resets the data in the tensor using a Hetu NDArray.
- get_data()->numpy.ndarray: Retrieves the data from the tensor and returns it as a numpy array. 
- get_device_group_union()->List[hetu.DeviceGroup]: Returns a list of hetu.DeviceGroup objects, where each DeviceGroup represents a group of devices within a specific pipeline.
- check_ds_hierarchy_equal(ds_hierarchy: List[hetu.DistributedStatesUnion])->bool: Checks if the tensor’s ds_hierarchy hierarchy is equal to a given ds_hierarchy
- check_ds_hierarchy_equal_except_split(ds_hierarchy: List[hetu.DistributedStatesUnion])->bool: Checks if the tensor’s ds_hierarchy is equal to the given ds_hierarchy, ignoring the data split dimension.
- get_or_compute()->hetu.NDArray: Retrieves the data or computes it if it has not already been computed.
- symbolic()->None: Initializes the symbolic shape of the tensor
- set_requires_grad(requires_grad:bool=false): Sets whether the tensor should track gradients for backpropagation.
- is_contiguous()->bool: checks whether the tensor is stored contiguously in memory.