### Init

#### hetu.voidified_initializer
```python
hetu.voidified_initializer()->hetu.Initializer()
```

This function returns a “voidified” initializer, which typically acts as a placeholder or default initializer. 

#### hetu.provided_initializer
```python
hetu.provided_initializer(provided_data:hetu.NDArray)->hetu.Initializer()
```

This function initializes a tensor using the provided data from an hetu.NDArray. 

#### hetu.constant_initializer
```python
hetu.constant_initializer(value:float)->hetu.Initializer()
```

This function creates an initializer that fills the tensor with a constant value. 

**Parameters**

- value (float): The constant value to initialize the tensor elements with.

#### hetu.ones_initializer
```python
hetu.ones_initializer()->hetu.Initializer()
```

This function creates an initializer that fills the tensor with ones. 

#### hetu.xavier_uniform_initializer
```python
hetu.xavier_uniform_initializer(gain:float=3.0)->hetu.Initializer()
```

This function creates an initializer based on the Xavier uniform initialization method.

**Parameters**

* gain (float, optional): A scaling factor to modify the variance of the uniform distribution. Default is 3.0.

#### hetu.xavier_normal_initializer
```python
hetu.xavier_normal_initializer(gain:float=1.0)->hetu.Initializer()
```

This function creates an initializer based on the Xavier normal initialization method.

**Parameters**

* gain (float, optional): A scaling factor to modify the variance of the normal distribution. Default is 1.0.

#### hetu.he_uniform_initializer
```python
hetu.he_uniform_initializer(gain:float=6.0)->hetu.Initializer()
```

**Parameters**

* gain (float, optional): A scaling factor to modify the variance of the uniform distribution. Default is 6.0.

#### hetu.he_normal_initializer
```python
hetu.he_normal_initializer(gain:float=2.0)->hetu.Initializer()
```

**Parameters**

* gain (float, optional): A scaling factor to modify the variance of the normal distribution. Default is 2.0.

#### hetu.lecun_uniform_initializer
```python
hetu.lecun_uniform_initializer(gain:float=3.0)->hetu.Initializer()
```

This function creates an initializer based on the LeCun uniform initialization method.

**Parameters**

* gain (float, optional): A scaling factor to modify the variance of the uniform distribution. Default is 3.0.

#### hetu.lecun_normal_initializer
```python
hetu.lecun_normal_initializer(gain:float=1.0)->hetu.Initializer()
```

This function creates an initializer based on the LeCun normal initialization method.

**Parameters**

* gain (float, optional): A scaling factor to modify the variance of the normal distribution. Default is 1.0.
