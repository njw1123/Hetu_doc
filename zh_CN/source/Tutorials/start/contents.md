### 快速入门

谢谢你选择Hetu！现在我们以一个CNN模型的训练来走进Hetu。在开始之前，请确保你已经正确地安装了Hetu。

一般地，深度学习任务包含几个重要步骤：1.数据集加载；2.构建模型；3.模型训练；4.模型验证；5.模型保存；6.模型加载；7.模型测试。下面你可以使用Hetu的API，来一步步实现上述步骤。

#### 模型示例

##### **导入Hetu**

新建一个python文件，并导入Hetu以及相关的模块。

```python
import hetu as ht
from hetu import dataloader as dl
from hetu import initializers as init
import numpy as np
```

 

##### **加载数据集**

Hetu内置了数据集的加载和切分batch的方法，你可以使用以下方法加载mnist数据集。

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

​		

##### **构建模型**

通过Hetu下的gpu_ops一步步将网络构建起来，Hetu内置了一些常用的初始化工具，你可以方便地进行模型初始化工作。

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

##### **模型训练**

模型的训练需要定义好Executor，然后直接使用Executor的run方法即可进行训练，并自动按照配置的优化器进行梯度的更新。

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

 

##### **模型验证**

你可以使用如下方法，进行模型的验证，并在验证集上评估模型的精度。

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

##### **结果输出**

对于以上CNN的模型训练和推理，在训练完成之后，你会得到如下的结果，从结果中可以看出，在10个epochs内，模型训练loss不断下降，同时模型在验证集上的精确呈上升趋势。

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



##### **模型保存**

训练完成之后，你可以使用以下命令将模型进行保存：

```python
#saving
executor.save(file_path='model_saved_dir', file_name='CNN_model')
```



##### **模型加载**

你可以使用以下命令加载模型：

```python
#loading
executor.load(file_path='model_saved_dir', file_name='CNN_model')
```



##### **模型测试**

你需要导入测试集，并重新构建模型，并加载训练好的模型进行测试。

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
