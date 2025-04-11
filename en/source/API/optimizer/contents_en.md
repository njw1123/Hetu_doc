
# Optimizer

## Adam

```python
class hetu.AdamOptimizer(init_lr, max_lr, min_lr, lr_warmup_steps, lr_decay_steps, lr_decay_style, start_wd, end_wd, wd_incr_steps, wd_incr_style, beta1, beta2, eps):
```

This api is used to calculate the learning rate and weight decay and create an Adam optimizer。

**Parameters:**

- `init_lr(float)`: initial learning rate
- `max_lr (float)`: maximum learning rate
- `min_lr (float)`: minimum learning rate
- `lr_warmup_steps (int)`: number of warmup steps
- `lr_decay_steps (int)`: number of decay steps
- `lr_decay_style (str)`: decay style for learning rate
- `start_wd (float)`: initial weight decay
- `end_wd (float)`: final weight decay
- `wd_incr_steps (int)`: number of weight decay increment steps
- `wd_incr_style (str)`: weight decay increment style
- `beta1 (float)`: Adam optimizer’s first momentum decay parameter
- `beta2 (float)`: Adam optimizer’s second momentum decay parameter
- `eps (float)`: To prevent division by zero

**Methods:**

- `minimize(loss, var_list, grad_loss, name) -> hetu.Tensor`: This function is used to minimize the given loss tensor by computing the gradients with respect to the specified variables.
  - `loss(hetu.Tensor)`: The loss tensor that needs to be minimized.
  - `var_list(List[hetu.Tensor], optional)`: A list of hetu.Tensor objects representing the variables (model parameters) that will be updated during the optimization process.If not specified, the function will automatically select the variables that require gradients.
  - `grad_loss(hetu.Tensor, optional)`: A tensor representing the gradient of the loss with respect to the loss function. If not provided, the gradients are computed internally using automatic differentiation.
  - `name(str, default="")`: A name for the operation.

- `get_states(var) -> Dict[str, hetu.Tensor]`: This function retrieves the states associated with a given tensor variable.
  - `var(hetu.Tensor)`: The tensor variable for which you want to obtain the optimizer states.

- `set_states(var, state_name, value) -> None`: This function sets the optimizer states for a given tensor variable.
  - `var(hetu.Tensor)`: The tensor variable for which you want to set the optimizer state.
  - `state_name(str)`: The name of the optimizer state you want to set (e.g., “var_mean”,)
  - `value(numpy.ndarray)`: The value that you want to set for the specified optimizer state.

## SGD

```python
class hetu.SGDOptimizer(init_lr, max_lr, min_lr, lr_warmup_steps, lr_decay_steps, lr_decay_style, momentum, nesterov)
```

**Parameters:**

- `init_lr(float)`: initial learning rate
- `max_lr (float)`: maximum learning rate
- `min_lr (float)`: minimum learning rate
- `lr_warmup_steps (int)`: number of warmup steps
- `lr_decay_steps (int)`: number of decay steps
- `lr_decay_style (str)`: decay style for learning rate
- `momentum (float, default=0.0)`: The momentum factor for the SGD optimizer.
- `nesterov (bool, default=False)`: A boolean flag that determines whether to use Nesterov Accelerated Gradient (NAG) for optimization

**Properties:**

- `learning_rate(float)`: learning rate of sgd

**Methods:**

- `minimize(loss, var_list, grad_loss, name) -> hetu.Tensor`: This function is used to minimize the given loss tensor by computing the gradients with respect to the specified variables.
  - `loss(hetu.Tensor)`: The loss tensor that needs to be minimized.
  - `var_list(List[hetu.Tensor], optional)`: A list of hetu.Tensor objects representing the variables (model parameters) that will be updated during the optimization process.If not specified, the function will automatically select the variables that require gradients.
  - `grad_loss(hetu.Tensor, optional)`: A tensor representing the gradient of the loss with respect to the loss function. If not provided, the gradients are computed internally using automatic differentiation.
  - `name(str, default="")`: A name for the operation.

## GradScaler

```python
GradScaler(init_scale, growth_factor, backoff_factor, growth_interval, enabled)
```

**Parameters:**

- `init_scale(float, default=65536.0)`: The initial scale factor to be applied to gradients.
- `growth_factor(float, default=2.0)`: The factor by which the scale is multiplied when it is increased.
- `backoff_factor(float, default=0.5)`: The factor by which the scale is reduced when an overflow is detected.
- `growth_interval(int, default=2000)`: The number of iterations after which the scale should be increased
- `enabled(bool, default=True)`:A boolean flag to enable or disable the scaler.

**Methods:**

- `scale(output: hetu.Tensor) -> hetu.Tensor`: Scales the gradients of the given tensor to the appropriate precision, applying the current scale factor.
- `scale(outputs: List[hetu.Tensor]) -> List[hetu.Tensor]`: Scales the gradients of multiple tensors in a list. This method is used when gradients from multiple outputs need to be scaled simultaneously.
- `minimize(op:hetu.SGDOptimizer,loss:hetu.Tensor,var_list:List[hetu.Tensor], grad_loss:hetu.Tensor) -> hetu.Tensor`: Applies the scaled gradients during the optimization step.
- `Update(new_scale:float) -> None`: Updates the scale for next iteration.
