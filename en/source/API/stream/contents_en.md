### Stream

#### hetu.stream

```
class hetu.stream(device:str, stream_id:int)
```

**Parameters**

- device(str):The format is “hostname/device_type:device_index”. (e.g., "localhost/cuda", "cuda:0", "cpu")
- stream_id(int): An integer representing the stream ID

```
class hetu.stream(device:hetu.device, stream_id:int)
```

**Parameters**

- device(hetu.device): A hetu.device object that represents the target device where the stream will be created and executed. 
- stream_id(int):  An integer representing the stream ID

```
class hetu.stream(stream: hetu.stream)
```

**Parameters**

- stream(hetu.stream): An existing hetu.stream object that will be copied to create a new stream with the same properties (such as device and stream ID).

**Properties**

- device(hetu.device): Represents the device on which the stream is running., 
- device_type(str): The type of device associated with the stream, which can be “cpu”, “gpu”, or “undetermined”.
- device_index(int): The index of the device on which the stream is running. 
- stream_index(int): The unique identifier for the stream within a device.
- is_define(bool): A boolean flag indicating whether the stream is properly defined and initialized. 