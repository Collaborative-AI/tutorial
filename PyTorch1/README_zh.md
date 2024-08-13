# Pytorch

<h4 align="center">
    <p>
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Pytorch/README.md">English</a>
        <b>简体中文</b> |
    </p>
</h4>


## 目录
1. [简介](#简介)
2. [设置 PyTorch](#设置-pytorch)
    - [安装](#安装)
    - [验证安装](#验证安装)
3. [使用张量](#使用张量)
    - [创建张量](#创建张量)
    - [张量操作](#张量操作)
    - [将张量移动到 GPU](#将张量移动到-gpu)
4. [构建一个简单的神经网络](#构建一个简单的神经网络)
    - [定义模型](#定义模型)
    - [定义损失函数和优化器](#定义损失函数和优化器)
5. [训练模型](#训练模型)
6. [评估模型](#评估模型)
7. [保存和加载模型](#保存和加载模型)
8. [下一步](#下一步)

---

## 简介

本教程将引导您了解 PyTorch 的基础知识，PyTorch 是一个强大的 Python 深度学习库。我们将介绍如何设置 PyTorch、使用张量、构建一个简单的神经网络等等。通过本教程，您将对 PyTorch 有一个基础性的理解，能够构建和训练自己的模型。

## 设置 PyTorch

在开始使用 PyTorch 之前，您需要在系统上安装它。

### 安装

通过 pip 安装 PyTorch，运行以下命令：

```bash
pip install torch torchvision
```

### 验证安装

要验证 PyTorch 是否正确安装，请打开 Python Shell 并尝试导入它：

```python
import torch

print(torch.__version__)  # 这应该会打印出已安装的 PyTorch 版本
```

如果此命令运行没有错误，说明 PyTorch 已成功安装。

## 使用张量

张量是 PyTorch 中的基本构建块。类似于 NumPy 数组，张量用于在神经网络中构建和操作数据。

### 创建张量

```python
import torch

# 创建一个全为零的张量
x = torch.zeros(5, 3)
print(x)

# 创建一个填充随机值的张量
y = torch.rand(5, 3)
print(y)

# 从列表创建张量
z = torch.tensor([[1, 2], [3, 4]])
print(z)
```

### 张量操作

PyTorch 提供了广泛的操作来处理张量：

```python
# 加法
result = x + y
print(result)

# 原地加法
y.add_(x)
print(y)

# 矩阵乘法
result = torch.mm(z, z.t())
print(result)

# 元素乘法
result = x * y
print(result)
```

### 将张量移动到 GPU

如果您有可用的 GPU，可以将张量移动到 GPU 以加速计算：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = x.to(device)
    y = y.to(device)
    print(x + y)
```

## 构建一个简单的神经网络

接下来，我们将使用 PyTorch 的 `torch.nn` 模块构建一个基本的神经网络。

### 定义模型

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 个输入特征，128 个输出特征
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)   # 10 个输出类别

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
print(model)
```

### 定义损失函数和优化器

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # 结合 softmax 和交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降
```

## 训练模型

以下是一个使用模型、损失函数和优化器的简单训练循环：

```python
# 假设您有一个 DataLoader 提供批量的图像和标签
for epoch in range(5):  # 5 个训练轮次
    running_loss = 0.0
    for images, labels in train_loader:
        # 展平图像
        images = images.view(images.shape[0], -1)
        
        # 清空参数的梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
```

## 评估模型

训练结束后，评估模型在测试数据集上的性能：

```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.shape[0], -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'模型在测试图像上的准确率: {100 * correct / total} %')
```

## 保存和加载模型

您可以将训练好的模型保存到磁盘，并在之后加载它：

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
```

## 下一步

本教程介绍了 PyTorch 的基础知识。为了进一步加深理解，建议您探索以下主题：

- **自定义数据集和 DataLoader**：学习如何处理不同类型的数据。
- **高级神经网络架构**：尝试构建更复杂的模型，如卷积神经网络（CNN）或循环神经网络（RNN）。
- **迁移学习**：使用预训练模型完成图像分类等任务。
- **超参数调优**：尝试不同的学习率、批量大小和优化器。

您可以在 [PyTorch 官方网站](https://pytorch.org/tutorials/) 上找到更多教程和详细文档。

---

通过本教程，您将对 PyTorch 有一个良好的基础认识，能够构建和训练自己的深度学习模型。