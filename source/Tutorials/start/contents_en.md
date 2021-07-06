##  Quick Start

Thank you for your attention to Hetu！

Let's start with the training of the CNN model now.

Please ensure you have finished the installation process before we start.  

In general, a deep learning task includes the following steps: 

​	1) dataset loading;

​	2) model building; 

​	3) model training; 

​	4) model validation. 

Let's implement these steps with APIs in Hetu. 


### Model Example


#### **Import HetuML Libraries**

You could start with a new python file, please import related Hetu libraries:

```python
from hetu import ndarray
from hetu import gpu_ops as ad
from hetu import optimizer
from hetu import dataloader as dl
from hetu import initializers as init
from models.load_data import load_mnist_data, convert_to_one_hot
import numpy as np
```



#### **Data Loading**

Hetu has built-in methods for data loading. You could use the following methods to load datasets. 


```python
executor_ctx = ndarray.gpu(0)
#training setting
opt = optimizer.SGDOptimizer(learning_rate=0.01)
batch_size = 128
num_epochs = 10
#data loading
datasets = load_mnist_data("mnist.pkl.gz")
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
```

Then, we transform the input data into one-hot format:

```python
#convert data to one-hot format
def local_convert_to_one_hot(batch):
  return convert_to_one_hot(batch, 10)

print('Building model...')
#data preprocess
x = dl.dataloader_op([
  dl.Dataloader(train_set_x, batch_size, 'train'),
  dl.Dataloader(valid_set_x, batch_size, 'validate'),
  ])

y_ = dl.dataloader_op([
  dl.Dataloader(train_set_y, batch_size, 'train', local_convert_to_one_hot),
  dl.Dataloader(valid_set_y, batch_size, 'validate', local_convert_to_one_hot),
])
```



#### **Model Building**

Hetu supports easy-to-use Python APIs for models training and testing. First, you can build  modules for training by using operators in Hetu. 

```python
def conv_relu_avg(x, shape):
  weight = init.random_normal(shape=shape, stddev=0.1)
  x = ad.conv2d_op(x, weight, padding=2, stride=1)
  x = ad.relu_op(x)
  x = ad.avg_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
  return x

def fc(x, shape):
  weight = init.random_normal(shape=shape, stddev=0.1)
  bias = init.random_normal(shape=shape[-1:], stddev=0.1)
  x = ad.array_reshape_op(x, (-1, shape[0]))
  x = ad.matmul_op(x, weight)
  y = x + ad.broadcastto_op(bias, x)
  return y
```

Then, you could define objective function for your model:

```python
print('Building CNN model...')
z1 = ad.array_reshape_op(x, [-1, 1, 28, 28])
z2 = conv_relu_avg(z1, (32, 1, 5, 5))
z3 = conv_relu_avg(z2, (64, 32, 5, 5))
y = fc(z3, (7 * 7 * 64, 10))
loss = ad.softmaxcrossentropy_op(y, y_)
loss = ad.reduce_mean_op(loss, [0])
train_op = opt.minimize(loss)
```

#### **Model Training**

The model training use the RUN method of Executor, and update gradients automatically according to the configured optimizer.


```python
executor = ad.Executor([loss, y, train_op], ctx=executor_ctx, dataloader_name='train')
n_train_batches = executor.batch_num
val_executor = ad.Executor([loss, y, y_], ctx=executor_ctx, dataloader_name='validate', inference=True)
n_valid_batches = val_executor.batch_num
# training
print("Start training loop...")
for i in range(num_epochs):
  print("Epoch %d" % i)
  loss_all = 0
  for minibatch_index in range(n_train_batches):
    loss_val, predict_y, _ = executor.run()
    loss_val = loss_val.asnumpy()
    loss_all += loss_val * x.dataloaders['train'].last_batch_size
  loss_all /= len(train_set_x)
  print("Loss = %f" % loss_all,end='')
```

 

#### **Model Validation**

You can use the following methods to validate your model and evaluate the accuracy of the model on the validation set.

```python
#validating
correct_predictions = []
for minibatch_index in range(n_valid_batches):
  loss_val, valid_y_predicted, y_val = val_executor.run(convert_to_numpy_ret_vals=True)
  correct_prediction = np.equal(
    np.argmax(y_val, 1),
    np.argmax(valid_y_predicted, 1)).astype(np.float)
  correct_predictions.extend(correct_prediction)
accuracy = np.mean(correct_predictions)
print("\tValidation accuracy = %f" % accuracy)
```

#### **Output**

For the above CNN model training, after the completion of the training, you will get the following results.

```
Building model...
Building CNN model...
Start training loop...
Epoch 0
Loss = 0.710760 Validation accuracy = 0.909255
Epoch 1
Loss = 0.319837 Validation accuracy = 0.928786
Epoch 2
Loss = 0.262297 Validation accuracy = 0.939503
Epoch 3
Loss = 0.227060 Validation accuracy = 0.945913
Epoch 4
Loss = 0.201521 Validation accuracy = 0.951623
Epoch 5
Loss = 0.181668 Validation accuracy = 0.956130
Epoch 6
Loss = 0.165777 Validation accuracy = 0.960036
Epoch 7
Loss = 0.152779 Validation accuracy = 0.962139
Epoch 8
Loss = 0.141957 Validation accuracy = 0.964944
Epoch 9
Loss = 0.132787 Validation accuracy = 0.967849
```