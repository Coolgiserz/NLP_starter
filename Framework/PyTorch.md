# PyTorch入门

PyTorch 于 2017 年初推出，给深度学习社区带来了深远的影响。它是由 [Facebook AI 研发团队](https://research.fb.com/category/facebook-ai-research-fair/)开发的，已经被各行各业和各个学术领域采用。Pytorch的特点是并且使用起来很方便。

## 基本操作

运算的操作：functional

```python
torch.manual_seed(446)# 固定随机种子，每次随机操作都再现原来的结果

```

PyTorch和numpy非常相似，与numpy之间的转换很容易

## 安装

```
pip install torch torchvision
```

## PyTorch实现深度学习

### 单层神经网络

元素级运算、矩阵运算

什么是原地操作？如何实现？

reshape、resize_、view方法的区别是？什么情况下用什么？

### 使用矩阵乘法构建网络

矩阵乘法比按元素级相乘后求和效率高，为什么？线性代数库做了优化，**怎么优化？**

### 多层网络解决方案

### Pytorch中的网络架构

```python
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
                     )

# Define the loss
criterion = nn.NLLLoss()

### Run this to check your work
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
```





```python
# Define your network architecture here
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=784,out_features=256),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256,out_features=128),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128,out_features=64),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=64,out_features=10),
    torch.nn.LogSoftmax()
)

# Create the network, define the criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the network here
epochs = 30
for epoch in range(epochs):
    run_loss = 0
    for image, label in iter(trainloader):
        image = image.view(image.shape[0], -1)
        output = model.forward(image)
        loss = criterion(output, label)
        run_loss += loss.item()
        loss.backward()
        optim.step()
    else:
        print("Loss: ", run_loss)


```



### Fashion-MNIST解决方案

```python
import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define your network architecture here
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=784,out_features=256),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256,out_features=128),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128,out_features=64),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=64,out_features=10),
    torch.nn.LogSoftmax(dim=1)
)

# Create the network, define the criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the network here
epochs = 30
for epoch in range(epochs):
    run_loss = 0
    for image, label in iter(trainloader):
        image = image.view(image.shape[0], -1)
        optim.zero_grad()
        output = model.forward(image)
        loss = criterion(output, label)
        run_loss += loss.item()
        loss.backward()
        optim.step()
    else:
        print("Loss: ", run_loss)


 %matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper

# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Calculate the class probabilities (softmax) for img
ps = torch.exp(model.forward(img))
print("True label: ", labels[0])
# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
```

### 推理和验证

#### 将网络封装成类

```python
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
```

#### Dropout解决方案

用于解决过拟合的策略有早期停止、Dropout

Dropout随机禁用输入单元，强制网络共享权重信息。

在训练时通过Dropout防止过拟合，验证、测试时需要关闭Dropout模式

```python
model.eval() # 将模型设置为评价模式：关闭dropout
model.train() # 训练模式，开启dropout
```

```python
## Define your model with dropout added
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
```



```python
## Train your model with dropout, and monitor the training progress with the validation loss and accuracy
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()    
        running_loss += loss.item()
        
    else:
        ## Implement the validation pass and print out the validation accuracy
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()

            # validation pass
            for images, labels in testloader:
                test_log_ps = model(images)
                test_loss += criterion(test_log_ps, labels)
                test_ps = torch.exp(test_log_ps)
                top_p, top_class = test_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))

        model.train()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        print("Epoch: {}/{}..".format(e+1, epochs),
              "Training: {:.3f}".format(running_loss/len(trainloader)),
               "Testing: {:.3f}".format(test_loss/len(testloader)),
               "Test Accuracy: {:.3f}".format(accuracy/len(testloader))
             )
#         print(f'Accuracy: {accuracy.item()*100}%')
```

### 保存和加载模型

保存模型的参数及网络架构，这样可方便以后重新加载

##### 保存模型：

```python
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
```

##### 加载模型：

```python
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
```

### GPU

何使用 GPU 加速网络计算流程？

#### GPU Workspace

如何使用 GPU 加速网络计算流程。 准备大规模地训练模型并优化参数时，**启用** GPU

涉及的所有模型和数据都必须移到 GPU 设备上，要注意模型创建和训练流程中的相关移动代码。

```python
model.cuda()

model.cpu()

images.cuda()

images.cpu()

torch.device("cpu")
gpu = torch.device("cuda")
x.to(gpu)
x.gpu()
```

### 迁移学习

https://pytorch.org/docs/0.3.0/torchvision/models.html

```python
data_dir = 'PATH'

# Define transforms for the training data and testing data
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.5, 0.5, 0.5))
    ]
)

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.5, 0.5, 0.5))
    ]
)

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
model = models.densenet121(pretrained=True)
model
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
```



```python
use_cuda = [True, False]
for cuda in use_cuda:
    if cuda:
        model.cuda()
    else:
        model.cpu()
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


    for ii, (inputs, labels) in enumerate(trainloader):
        if cuda:           
            # Move input and label tensors to the GPU
            inputs, labels = inputs.cuda(), labels.cuda()
        else:
            inputs, labels = inputs.cpu(), labels.cpu()
        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
```
