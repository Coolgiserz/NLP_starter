# Tensorflow

## 案例实战

### Tensorflow实现卷积神经网络

如何使用tensorflow实现卷积层？池化层？组成卷积神经网络

#### Tensorflow卷积层

TensorFlow 提供了 [`tf.nn.conv2d()`](https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d) 和 [`tf.nn.bias_add()`](https://tensorflow.google.cn/api_docs/python/tf/nn/bias_add) 函数来创建你自己的卷积层。

```python
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

上述代码用了 [`tf.nn.conv2d()`](https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d) 函数来计算卷积，`weights` 作为滤波器，`[1, 2, 2, 1]` 作为 strides。TensorFlow 对每一个 `input` 维度使用一个单独的 stride 参数，`[batch, input_height, input_width, input_channels]`。我们通常把 `batch` 和 `input_channels` （`strides` 序列中的第一个第四个）的 stride 设为 `1`。

你可以专注于修改 `input_height` 和 `input_width`， `batch` 和 `input_channels` 都设置成 1。`input_height` 和 `input_width` strides 表示滤波器在`input` 上移动的步长。上述例子中，在 `input`之后，设置了一个 5x5 ，stride 为 2 的滤波器。

[`tf.nn.bias_add()`](https://tensorflow.google.cn/api_docs/python/tf/nn/bias_add) 函数对矩阵的最后一维加了偏置项。

### Tensorflow最大池化

TensorFlow 提供了 [`tf.nn.max_pool()`](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) 函数，用于对卷积层实现 [最大池化](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) 。

```python
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

[`tf.nn.max_pool()`](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) 函数实现最大池化时， `ksize`参数是滤波器大小，`strides`参数是步长。2x2 的滤波器配合 2x2 的步长是常用设定。

`ksize` 和 `strides` 参数也被构建为四个元素的列表，每个元素对应 input tensor 的一个维度 (`[batch, height, width, channels]`)，对 `ksize` 和 `strides` 来说，batch 和 channel 通常都设置成 `1`。

池化层总的来说是为了减小输出的数量，其同样可减少后续层中的参数，避免过拟合。

但近年来，池化层并不是很受青睐。部分原因是：

- 现在的数据集又大又复杂，我们更关心欠拟合问题。
- Dropout 是一个更好的正则化方法。
- 池化导致信息损失。想想最大池化的例子，*n* 个数字中我们只保留最大的，把余下的 *n-1* 完全舍弃了。

注意：池化层的输出深度与输入的深度相同。另外池化操作是分别应用到每一个深度切片层。

下图给你一个最大池化层如何工作的示例。这里，最大池化滤波器的大小是 2x2。当最大池化层在输入层滑动时，输出是这个 2x2 方块的最大值。



```python
input = tf.placeholder(tf.float32, (None, 4, 4, 5))
filter_shape = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = 'VALID'
pool = tf.nn.max_pool(input, filter_shape, strides, padding)
```



### 1*1卷积

在卷积中散步1*1卷积可让模型变得更深（有更多参数），并且不需要改变神经网络结构。

### Inception

不局限于单个卷积运算，而是将多个模块组合，如平均池化后接$1*1$卷积；$1*1$卷积接着$3*3$的卷积

### Tensorflow中的卷积网络

网络的结构跟经典的 CNNs 结构一样：是卷积层，最大池化层和全链接层的组合。

这里我们导入 MNIST 数据集，用一个方便的函数完成对数据集的 batch，scale 和 One-Hot编码。

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# Parameters
# 参数
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
# 用来验证和计算准确率的样本数
# 如果内存不够，可以调小这个数字
test_valid_size = 256

# Network Parameters
# 神经网络参数
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
```

#### 权重和偏置

```python
# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}
```

#### 卷积

```python
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
```

#### 最大池化

```python
def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')
```

#### 模型

```python
def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer - 7*7*64 to 1024
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
```

#### Session

```python
# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf. global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
```

你应该看一下[二维卷积的文档](https://www.tensorflow.org/api_guides/python/nn#Convolution)。文档大部分都很清楚，padding这一部分，可能会有点难以理解。`padding`会根据你给出的 `'VALID'` 或者 `'SAME'` 参数，做相应改变。

这些也是需要回顾的：

1. TensorFlow 变量。

2. Truncated 正态分布 - 在 TensorFlow 中你需要在一个正态分布的区间中初始化你的权值。

3. 根据输入大小、滤波器大小，来决定输出维度（如下所示）。你用这个来决定滤波器应该是什么样：

   ```
    new_height = (input_height - filter_height + 2 * P)/S + 1
    new_width = (input_width - filter_width + 2 * P)/S + 1
   ```

```python
"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # TODO: Define the filter weights `F_W` and filter bias `F_b`.
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    F_W = tf.Variable(tf.random_normal([2,2,1,3]))
    F_b = tf.Variable(tf.random_normal([1]))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'SAME'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)
```

Tensorflow池化层

```python
"""
Set the values to `strides` and `ksize` such that
the output shape after pooling is (1, 2, 2, 1).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.max_pool` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

def maxpool(input):
    # TODO: Set the ksize (filter size) for each dimension (batch_size, height, width, depth)
    ksize = [1, 2, 2, 1]
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#max_pool
    return tf.nn.max_pool(input, ksize, strides, padding)
    
out = maxpool(X)
```

#### CNNs补充材料

- [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com)
- [deep learning](http://www.deeplearningbook.org)

## 可视化工具：Tensorboard

```
tensorboard --logdir=PATH
```

