### Symbol

```
class hetu.IntSymbol(val: int)
```

The constructor for creating an IntSymbol object, which stores an integer value. This class can be used to represent symbolic values in computations, where the integer value is encapsulated as part of a symbolic expression

**Parameters**

- val(int): The integer value to be stored in the IntSymbol.

**Properties**

- is_leaf(bool): A boolean property that indicates whether the IntSymbol is a leaf node in a computation graph.
- data(int): The integer data stored within the IntSymbol object.

**methods**

- get_data()->int:Retrieves the current value stored in the IntSymbol.
- set_data(data:int)->None:Sets the value of the IntSymbol to the specified integer.
- reset_data(data:int)->None:Resets the value of the IntSymbol to the specified integer

**NOTE:** IntSymbol also supports common operations with other IntSymbol objects and integers.

```
hetu.add(x:hetu.IntSymbol, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.add(x:hetu.IntSymbol, y:int)->hetu.IntSymbol
hetu.add(x:int, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.sub(x:hetu.IntSymbol, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.sub(x:hetu.IntSymbol, y:int)->hetu.IntSymbol
hetu.sub(x:int, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.mul(x:hetu.IntSymbol, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.mul(x:hetu.IntSymbol, y:int)->hetu.IntSymbol
hetu.mul(x:int, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.div(x:hetu.IntSymbol, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.div(x:hetu.IntSymbol, y:int)->hetu.IntSymbol
hetu.div(x:int, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.rem(x:hetu.IntSymbol, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.rem(x:hetu.IntSymbol, y:int)->hetu.IntSymbol
hetu.rem(x:int, y:hetu.IntSymbol)->hetu.IntSymbol
hetu.neg(x:hetu.IntSymbol)->hetu.IntSymbol
```