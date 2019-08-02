---
layout:     post
title:      PyTorch-简单CNN实现
subtitle:   构建分类CNN
date:       2019-04-20
author:     zhouchen
header-img: img/post-bg-pytorch.png
catalog: true
tags:
    - PyTorch
---

# 简单实现CNN
- 本案例实现基本的CNN并用Mnist数据集测试网络效果
- 步骤
	- 导入数据
		- torch的数据在torchvision中
		- torchvision处理了很多经典数据集，使用接口下载即可
		- 代码
			- code```python
				import torch
				import torch.nn as nn
				from torch.utils.data import DataLoader
				import torchvision
				import matplotlib.pyplot as plt
				torch.manual_seed(2019)
				
				# 使用著名的Mnist手写数据
				train_data = torchvision.datasets.MNIST(root='data/',  # 数据集下载位置
				                                        train=True,  # 数据集是是不是训练集
				                                        transform=torchvision.transforms.ToTensor(),  # 数据集数值标准化
				                                        download =False,  # 第一使用需要设置为True，后续下载过设置为False
				                                       )
				test_data = torchvision.datasets.MNIST(root='data/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
				```
	- 定义网络结构
		- CNN一般由卷积层、池化层、标准化层组成
		- 代码
			- ```python
				# 定义网络结构
				class CNN(nn.Module):
				    def __init__(self):
				        super(CNN, self).__init__()
				        self.conv1 = nn.Sequential(
				            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),  # 初始化卷积核大小为3*3
				            nn.ReLU(),  # 使用relu激活
				            nn.MaxPool2d(kernel_size=2),  # 采用最大池化2*2
				        )
				        self.conv2 = nn.Sequential(
				            nn.Conv2d(16, 32, 3, 1, 1), 
				            nn.ReLU(), 
				            nn.MaxPool2d(kernel_size=2),
				        )
				        
				        self.output = nn.Linear(32 * 7 * 7, 10)  # 使用全连接输出
				        
				    def forward(self, x):
				        x = self.conv1(x)
				        x = self.conv2(x)
				        x = x.view(x.size(0), -1)
				        output = self.output(x)
				        return output
				    
				cnn = CNN()
				print(cnn)
				```
	- 神经网络训练，和一般训练一致
		- 代码
			- ```python
				# 训练过程
				optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) 
				loss_func = nn.CrossEntropyLoss()
				for epoch in range(EPOCH):
				    for step, (b_x, b_y) in enumerate(train_loader): 
				        output = cnn(b_x) 
				        loss = loss_func(output, b_y) 
				        optimizer.zero_grad()
				        loss.backward()
				        optimizer.step()
				```
	- 验证在验证集上的效果
		- 运行结果
			- ![](https://img-blog.csdnimg.cn/201904221918547.png)
- 补充说明
	- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体完整代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）