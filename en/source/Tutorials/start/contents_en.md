##  Quick Start

Thank you for your attention to Hetu！

Let's start with the training of the CNN model now.

Please ensure you have finished the installation process before we start.  

In general, a deep learning task includes the following steps: 

​	1) dataset loading;

​	2) model building; 

​	3) model training; 

​	4) model validation. 

​	5) model saving. 

​	6) model loading. 

​	7) model testing. 

Let's implement these steps with APIs in Hetu. 


### Model Example


#### **Import HetuML Libraries**

You could start with a new python file, please import related Hetu libraries:

```python
import hetu as ht
from hetu import dataloader as dl
from hetu import initializers as init
import numpy as np
```



#### **Data Loading**

Hetu has built-in methods for data loading. You could use the following methods to load datasets. 


```python
executor_ctx = ht.gpu(0)
opt = ht.optim.SGDOptimizer(learning_rate=0.01)

batch_size = 128
num_epochs = 10

#data loading
print('Loading datasets...')
datasets = ht.data.mnist()
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

x = dl.dataloader_op([
  dl.Dataloader(train_set_x, batch_size, 'train'),
  dl.Dataloader(valid_set_x, batch_size, 'validate'),
  ])

y_ = dl.dataloader_op([
  dl.Dataloader(train_set_y, batch_size, 'train'),
  dl.Dataloader(valid_set_y, batch_size, 'validate'),
])
```



#### **Model Building**

Hetu supports easy-to-use Python APIs for models training and testing. First, you can build  modules for training by using operators in Hetu. 

```python
#model building
print('Building model...')

def conv_relu_avg(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    x = ht.conv2d_op(x, weight, padding=2, stride=1)
    x = ht.relu_op(x)
    x = ht.avg_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def fc(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    bias = init.random_normal(shape=shape[-1:], stddev=0.1)
    x = ht.array_reshape_op(x, [-1, shape[0]])
    x = ht.matmul_op(x, weight)
    y = x + ht.broadcastto_op(bias, x)
    return y


x = ht.array_reshape_op(x, [-1, 1, 28, 28])
x = conv_relu_avg(x, [32, 1, 5, 5])
x = conv_relu_avg(x, [64, 32, 5, 5])
y = fc(x, [7 * 7 * 64, 10])
loss = ht.softmaxcrossentropy_op(y, y_)
loss = ht.reduce_mean_op(loss, [0])
train_op = opt.minimize(loss)
```

#### **Model Training**

The model training use the RUN method of Executor, and update gradients automatically according to the configured optimizer.


```python
eval_nodes = {'train': [loss, y, y_, train_op], 'validate': [loss, y, y_]}
executor = ht.Executor(eval_nodes, ctx=executor_ctx)
n_train_batches = executor.get_batch_num('train')
n_valid_batches = executor.get_batch_num('validate')

# training
print("Start training loop...")
running_time = 0
for i in range(num_epochs + 1):
    print("Epoch %d" % i)
    loss_all = 0
    batch_num = 0
    correct_predictions = []
    for minibatch_index in range(n_train_batches):
        loss_val, predict_y, y_val, _ = executor.run('train', eval_node_list=eval_nodes['train'], convert_to_numpy_ret_vals=True)
        loss_all += loss_val
        batch_num += 1
        correct_prediction = np.equal(
            np.argmax(y_val, 1),
            np.argmax(predict_y, 1)).astype(np.float32)
        correct_predictions.extend(correct_prediction)

    loss_all /= batch_num
    accuracy = np.mean(correct_predictions)
    print("Train loss = %f Train accuracy = %f" % (loss_all, accuracy))
```

 

#### **Model Validation**

You can use the following methods to validate your model and evaluate the accuracy of the model on the validation set.

```python
val_loss_all = 0
batch_num = 0
correct_predictions = []
for minibatch_index in range(n_valid_batches):
    loss_val, valid_y_predicted, y_val = executor.run('validate', eval_node_list=eval_nodes['validate'], convert_to_numpy_ret_vals=True)
    val_loss_all += loss_val
    batch_num += 1
    correct_prediction = np.equal(
        np.argmax(y_val, 1),
        np.argmax(valid_y_predicted, 1)).astype(np.float32)
    correct_predictions.extend(correct_prediction)

val_loss_all /= batch_num
accuracy = np.mean(correct_predictions)
print("Validation loss = %f Validation accuracy = %f" % (val_loss_all, accuracy))
```

#### **Output**

For the above CNN model training, after the completion of the training, you will get the following results.

```
Loading datasets...
Building model...
Start training loop...
Epoch 0
Train loss = 0.790133 Train accuracy = 0.762640
Validation loss = 0.337973 Validation accuracy = 0.902845
Epoch 1
Train loss = 0.324640 Train accuracy = 0.908454
Validation loss = 0.257370 Validation accuracy = 0.928586
Epoch 2
Train loss = 0.259951 Train accuracy = 0.926623
Validation loss = 0.218450 Validation accuracy = 0.939904
Epoch 3
Train loss = 0.222016 Train accuracy = 0.937420
Validation loss = 0.192066 Validation accuracy = 0.947716
Epoch 4
Train loss = 0.195031 Train accuracy = 0.945713
Validation loss = 0.171819 Validation accuracy = 0.953425
Epoch 5
Train loss = 0.174385 Train accuracy = 0.951482
Validation loss = 0.155746 Validation accuracy = 0.957332
Epoch 6
Train loss = 0.158009 Train accuracy = 0.955789
Validation loss = 0.142644 Validation accuracy = 0.961739
Epoch 7
Train loss = 0.144659 Train accuracy = 0.959756
Validation loss = 0.131887 Validation accuracy = 0.964443
Epoch 8
Train loss = 0.133586 Train accuracy = 0.963181
Validation loss = 0.122931 Validation accuracy = 0.966346
Epoch 9
Train loss = 0.124299 Train accuracy = 0.965645
Validation loss = 0.115433 Validation accuracy = 0.968650
Epoch 10
Train loss = 0.116425 Train accuracy = 0.967548
Validation loss = 0.109135 Validation accuracy = 0.970553
```

#### **Model Saving**

After training, you can save the model by using the following methods.

```python
#saving
executor.save(file_path='model_saved_dir', file_name='CNN_model')
```


#### **Model Loading**

Also, You can load the saved model to do testing.

```python
#loading
executor.load(file_path='model_saved_dir', file_name='CNN_model')
```


#### **Model Testing**

You need to rebuild the model and load the saved parameter like above. Then, you can complete the testing. 


```python
import hetu as ht
from hetu import dataloader as dl
from hetu import initializers as init
import numpy as np

executor_ctx = ht.gpu(0)
batch_size = 128

#data loading
print('Loading datasets...')
datasets = ht.data.mnist()
test_set_x, test_set_y = datasets[2]

x = dl.dataloader_op([
  dl.Dataloader(test_set_x, batch_size, 'default')
  ])

y_ = dl.dataloader_op([
  dl.Dataloader(test_set_y, batch_size, 'default')
  ])
#model building
print('Building model...')

def conv_relu_avg(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    x = ht.conv2d_op(x, weight, padding=2, stride=1)
    x = ht.relu_op(x)
    x = ht.avg_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x

def fc(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    bias = init.random_normal(shape=shape[-1:], stddev=0.1)
    x = ht.array_reshape_op(x, [-1, shape[0]])
    x = ht.matmul_op(x, weight)
    y = x + ht.broadcastto_op(bias, x)
    return y

x = ht.array_reshape_op(x, [-1, 1, 28, 28])
x = conv_relu_avg(x, [32, 1, 5, 5])
x = conv_relu_avg(x, [64, 32, 5, 5])
y = fc(x, [7 * 7 * 64, 10])
loss = ht.softmaxcrossentropy_op(y, y_)
loss = ht.reduce_mean_op(loss, [0])

executor = ht.Executor([loss, y, y_], ctx=executor_ctx)
n_test_batches = executor.batch_num
executor.load(file_path='model_saved_dir', file_name='CNN_model')

# test
print("Start testing...")
correct_predictions = []
for minibatch_index in range(n_test_batches):
    loss_val, test_y_predicted, y_val = executor.run(convert_to_numpy_ret_vals=True)
    correct_prediction = np.equal(
            np.argmax(y_val, 1),
            np.argmax(test_y_predicted, 1)).astype(np.float32)
    correct_predictions.extend(correct_prediction)
accuracy = np.mean(correct_predictions)
print("Test accuracy = %f" % (accuracy))
```