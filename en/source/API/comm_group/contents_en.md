
### Comm_group

#### hetu.init_comm_group

```python
hetu.init_comm_group(device_num:int, device_idxs:List[int], server_address:str)->hetu.device
```

This function initializes a communication group for distributed training or parallel computing. 

**Parameters**

- device_num (int):The number of devices to be included in the communication group.
- device_idxs (List[int]): A list of integers representing a mapping from sequential device IDs (e.g., 0, 1, 2, …) to custom device IDs
- server_address (str):The address of the server that the devices will communicate with. 

**Returns**

- A hetu.device object representing local device

#### hetu.local_device

```python
hetu.local_device()->hetu.device
```

This function returns the local device on which the computation will be performed. 

**Returns**

- A hetu.device object representing the local device

#### hetu.global_device_group

```python
hetu.global_device_group()->hetu.DeviceGroup
```

This function returns a group of all global devices involved in the computation or distributed training.

**Returns**

- A hetu.DeviceGroup object containing all devices

#### hetu.global_comm_barrier_rpc

```python
hetu.global_comm_barrier_rpc()->None
```

This function sets a communication barrier in a distributed system using Remote Procedure Calls (RPC). It ensures that all devices in the communication group reach this point before proceeding, synchronizing the devices.
    
#### hetu.global_comm_barrier_mpi

```python
hetu.global_comm_barrier_mpi()->None
```

This function sets a communication barrier in a distributed system using Message Passing Interface (MPI). It ensures that all devices in the communication group synchronize at this barrier point.