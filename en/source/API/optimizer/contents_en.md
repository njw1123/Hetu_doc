
### Optimizer
#### Adam

```
class hetu.AdamOptimizer(init_lr:float, max_lr:float, min_lr:float, lr_warmup_steps:int, lr_decay_steps:int,  lr_decay_style:string, start_wd:float, end_wd:float, wd_incr_steps:int, wd_incr_style:string, beta1:float, beta2:float, eps:float):

```

This api is used to calculate the learning rate and weight decay and create an Adam optimizer。

**Parameters**

-  init_lr(float)： initial learning rate
- max_lr (float)：maximum learning rate
- min_lr (float)：minimum learning rate
- lr_warmup_steps (int)：number of warmup steps
- lr_decay_steps (int)：number of decay steps
- lr_decay_style (str)：decay style for learning rate
- start_wd (float)：initial weight decay
- end_wd (float)：final weight decay
- wd_incr_steps (int)：number of weight decay increment steps
- wd_incr_style (str)：weight decay increment style
- beta1：Adam optimizer’s first momentum decay parameter
- beta2：Adam optimizer’s second momentum decay parameter
- eps：To prevent division by zero

**methods**

- minimize(loss:hetu.Tensor, var_list:List[hetu.Tensor]=None, grad_loss:hetu.Tensor=None, name:str="")->hetu.Tensor：This function is used to minimize the given loss tensor by computing the gradients with respect to the specified variables.
  - loss(hetu.Tensor): The loss tensor that needs to be minimized. 
  - var_list(List[hetu.Tensor]): A list of hetu.Tensor objects representing the variables (model parameters) that will be updated during the optimization process.If not specified, the function will automatically select the variables that require gradients. 
  - grad_loss(hetu.Tensor): A tensor representing the gradient of the loss with respect to the loss function. If not provided, the gradients are computed internally using automatic differentiation.
  - name(str): A name for the operation. 

- get_states(var:hetu.Tensor)->Dict[str, hetu.Tensor]：This function retrieves the states associated with a given tensor variable.
  - var(hetu.Tensor): The tensor variable for which you want to obtain the optimizer states. 
- set_states(var:hetu.Tensor, state_name:str, value:numpy.ndarray)->None:This function sets the optimizer states for a given tensor variable.
  - var(hetu.Tensor): The tensor variable for which you want to set the optimizer state. 
  - state_name(str): The name of the optimizer state you want to set (e.g., “var_mean”,)
  - value(numpy.ndarray):  The value that you want to set for the specified optimizer state.

#### SGD

```
class hetu.SGDOptimizer(init_lr: float, max_lr: float, min_lr: float, lr_warmup_steps: int, lr_decay_steps: int, lr_decay_style: str, momentum: float = 0.0, nesterov: bool = False)
```

**Parameters**

- init_lr(float)： initial learning rate
- max_lr (float)：maximum learning rate
- min_lr (float)：minimum learning rate
- lr_warmup_steps (int)：number of warmup steps
- lr_decay_steps (int)：number of decay steps
- lr_decay_style (str)：decay style for learning rate
- momentum (float, optional): The momentum factor for the SGD optimizer.
- nesterov (bool, optional): A boolean flag that determines whether to use Nesterov Accelerated Gradient (NAG) for optimization

**Properties**

- learning_rate(float): learning rate of sgd

**methods**

- minimize(loss:hetu.Tensor, var_list:List[hetu.Tensor]=None, grad_loss:hetu.Tensor=None, name:str="")->hetu.Tensor：This function is used to minimize the given loss tensor by computing the gradients with respect to the specified variables.
  - loss(hetu.Tensor): The loss tensor that needs to be minimized. 
  - var_list(List[hetu.Tensor]): A list of hetu.Tensor objects representing the variables (model parameters) that will be updated during the optimization process.If not specified, the function will automatically select the variables that require gradients. 
  - grad_loss(hetu.Tensor): A tensor representing the gradient of the loss with respect to the loss function. If not provided, the gradients are computed internally using automatic differentiation.
  - name(str): A name for the operation. 

#### GradScaler

```
GradScaler(init_scale:float=65536.0, growth_factor:float=2.0,  backoff_factor:float=0.5, growth_interval:int=2000, enabled:bool=True)
```

**Parameters**

- init_scale(float): The initial scale factor to be applied to gradients.
- growth_factor(float): The factor by which the scale is multiplied when it is increased. 
- backoff_factor(float): The factor by which the scale is reduced when an overflow is detected 
- growth_interval(int): The number of iterations after which the scale should be increased
- enabled(bool):A boolean flag to enable or disable the scaler.

**methods**

- scale(output: hetu.Tensor)->hetu.Tensor: Scales the gradients of the given tensor to the appropriate precision, applying the current scale factor.
- scale(outputs: List[hetu.Tensor])->List[hetu.Tensor]: Scales the gradients of multiple tensors in a list. This method is used when gradients from multiple outputs need to be scaled simultaneously.
- minimize(op:hetu.SGDOptimizer,loss:hetu.Tensor,var_list:List[hetu.Tensor], grad_loss:hetu.Tensor)->hetu.Tensor: Applies the scaled gradients during the optimization step. 
- Update(new_scale:float)->None: Updates the scale for next iteration.
