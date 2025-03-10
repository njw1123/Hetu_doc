### Device

#### hetu.device

```
class hetu.device(device: str, multiplex:int)
```

**Parameters**

- device (str): The format is “hostname/device_type:device_index”. (e.g., "localhost/cuda", "cuda:0", "cpu")
- multiplex (int): The number of tasks that can run concurrently on the same device.

```
class hetu.Device(device: hetu.device)
```

**Parameters**

- device (hetu.device): An already defined hetu.device object.

**Properties**

- type (str): Returns the device type, which can be “cpu”, “gpu”, or “undetermined”.
- index (int): The device index. In distributed training with multiple “gpus”, each “gpu” is assigned a unique index.
- is_cpu (bool): Indicates whether the device is a “cpu”.
- is_undetermined (bool): Indicates whether the device has not been explicitly specified.
- local (bool): Indicates whether the device is a local device.
- hostname (str): The hostname of the device.
- multiplex (int): The maximum number of tasks that can run concurrently on the device.

**Methods**

- get_local_hostname() -> str: Returns the hostname of the device.

#### hetu.DeviceGroup

```
class hetu.DeviceGroup(device: List[str])
```

**Parameters**

- device(List[str]): A list where each string refers to a device in the format “hostname/device_type:device_index”. (e.g., ["localhost/cuda", "cuda:0", "cpu"])

```
class hetu.DeviceGroup(device_group: hetu.DeviceGroup)
```

**Parameters**

- device_group (hetu.DeviceGroup): An already defined hetu.DeviceGroup object.

**Properties**

- empty (bool): Indicates whether the device group contains valid devices.
- num_devices (int): The total number of devices in the group.
- device_index (List[int]): Returns a list of indices for each device in the DeviceGroup.

**Methods**

- contains(device: hetu.device) -> bool: Checks if a specific device is part of the DeviceGroup.
- contains(device: str, multiplex: int) -> bool: Checks if a device with the specified name and multiplex is part of the DeviceGroup.
- get(index: int) -> hetu.device: Returns the device at the specified index in the DeviceGroup.
- get_index(hetu.device) -> int: Returns the index of the specified device in the DeviceGroup.
