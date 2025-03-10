### NDArray
#### hetu.NDArray

```
class hetu.NDArray(data:numpy.ndarray, dtype:hetu.dtype=None, device:hetu.device=None)
```

**Parameters**

- data (numpy.ndarray): A numpy ndarray object containing the data that will be used to initialize the NDArray. 
- dtype (hetu.dtype, optional): The desired data type for the new NDArray. If not specified, the data type will be inferred from the data parameter 
- device (hetu.device, optional): The target device for the NDArray (such as CPU or GPU). 

```
class hetu.NDArray(data:hetu.NDArray, dtype:hetu.dtype=None, device:hetu.device=None)
```

**Parameters**

- data (hetu.NDArray): An existing hetu.NDArray object that will be used as the source for initializing the new NDArray. 
- dtype (hetu.dtype, optional): The desired data type for the new NDArray. If not specified, the data type will be inferred from the data parameter 
- device (hetu.device, optional): The target device for the NDArray (such as CPU or GPU). 

```
class hetu.NDArray(data, dtype:hetu.dtype=None, device:hetu.device=None)
```

- data：A generic Python object (e.g., list, tuple, or any sequence-like object) that will be converted into an NDArray.
- dtype (hetu.dtype, optional): The desired data type for the new NDArray. If not specified, the data type will be inferred from the data parameter 
- device (hetu.device, optional): The target device for the NDArray (such as CPU or GPU). 


**Properties**

- device(hetu.device): The device on which the NDArray is stored.
- dtype(hetu.dtype): The data type of the elements in the NDArray. 
- shape(List[int])：A list representing the shape of the NDArray. 
- data_ptr(int)：A pointer to the raw data of the NDArray. 


**Methods**

- numpy(force:bool, save:bool)->numpy.ndarray: Converts the NDArray to a numpy ndarray. 
  - force (bool): If True, and the NDArray is not on the CPU, the method will force the transfer of the data to the CPU before converting it into a numpy ndarray.
  - save (bool): If False, for float16 and bfloat16 data types, the original data is preserved as int16_t. If True, the data is converted to float32 for these types to ensure compatibility with numpy arrays
- numel()->int: Returns the total number of elements in the NDArray.
- to(dtype:hetu.dtype)->hetu.NDArray: Converts the NDArray to the specified data type
- to(device: hetu.device, dtype: hetu.dtype)->hetu.NDArray: Converts the NDArray to a specified device and data type.
- to(other: hetu.NDArray)->hetu.NDArray: Converts the NDArray to have the same device and data type as another NDArray.
- slice(begin_pos List[int], output_shape List[int])->hetu.NDArray: Slices the NDArray according to the specified beginning position and output shape.
  - begin_pos (List[int]): A list of integers specifying the starting position for each dimension.
  - output_shape (List[int]): A list of integers specifying the shape of the resulting slice.
- copy()->hetu.NDArray: Creates a copy of the NDArray.
- transpose(perms:List[int])->hetu.NDArray: Transposes the NDArray according to the specified permutation of axes.
  - perms (List[int]): A list of integers specifying the new order of axes.
- view(shape:list[int])->hetu.NDarray: Creates a view of the NDArray with the specified shape. This does not copy the data, but rather creates a new view of the original data with the given shape.
  - shape (List[int]): A list of integers specifying the desired shape.

**Class methods**

- numpy_to_NDArray(data: numpy.ndarray)->hetu.NDArray: Converts a numpy array (numpy.ndarray) into a Hetu NDArray. 
- numpy_to_NDArray(data: numpy.ndarray, dynamic_shape:List[int])->hetu.NDArray: Converts a numpy array (numpy.ndarray) into a Hetu NDArray, while specifying a dynamic shape (i.e., reshaping the array) during the conversion.
- numpy_to_NDArray(data: numpy.ndarray, dtype:hetu.dtype)->hetu.NDArray: Converts a numpy array (numpy.ndarray) into a Hetu NDArray, while specifying the desired data type (dtype) for the new NDArray. 
