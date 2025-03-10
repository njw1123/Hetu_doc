### Profiler

Hetu provides four types of time analysis:

1. **op_view**: Records the execution time of each operator.
2. **optype_view**: Records execution time categorized by operator type.
3. **optype_with_inputs_view**: Records execution time categorized by operator type and input shape/size.
4. **graph_view**: Provides an overview of execution, including total run time, forward computation, backward computation, pipeline communication (pp-p2p), etc.

**Example:** 

```python
import hetu

with hetu.profiler(enabled=True, record_shapes=True) as profiler:
    # Training code
    print(profiler.summary())
```

**Parameters:**
- enabled: Whether to enable the profiler.
- record_shapes: Whether to classify operators based on input shapes for time statistics.

**Methods:**
- summary(): Returns the profiling results as a dictionary.
