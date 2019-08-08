---
title:      PyTorch实现分类网络
date:       2019-04-18
categories: 'PyTorch'
description: 演示PyTorch中实现分类模型的方法。
tags:
    - 深度学习
    - PyTorch
updated: 
music-id: 
---
## 简介
- 本案例使用只含有一个隐层20个神经元的神经网络进行分类，并显示分类过程。


## 步骤
- 创建数据集
	- 方便演示，使用空间分布差异明显的两类数据集。（二分类）
	- 代码
		- ```python
			import torch
			import matplotlib.pyplot as plt
			%matplotlib inline
			
			# 创建分布明显区分的两块数据点
			n_data = torch.ones(100, 2)
			
			# 上分布数据和下分布数据都是正太分布
			x0 = torch.normal(2*n_data, 1)  # 二维特征
			y0 = torch.zeros(100)  # 一维目标0
			x1 = torch.normal(-2*n_data, 1)  # 二维目标
			y1 = torch.ones(100)  # 一维特征1
			
			# 经过上面数据集构建，显然这是一个0-1分布
			
			# 合并数据
			x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 合并为(200,2)的维度
			y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # 合并为(200, [1])的维度
			print(x.shape)
			print(y.shape)
			
			plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=100, lw=0)
			plt.show()
			```
- 搭建网络结构
	- 不同于之前不同结构的堆砌，这次使用了类似Keras的Sequential的方式，使网络结构更加清晰。
	- 代码
		- ```python
			# 构建单隐层网络，不同于之前回归使用继承类的方式，这里使用类似Keras的add方式
			from torch.nn import Sequential, Linear, Module, ReLU
			import torch.nn.functional as F
			class Net(Module):
				def __init__(self):
					super(Net, self).__init__()
					self.layer = Sequential(
						Linear(2, 10),
						ReLU(),
						Linear(10, 2),
					)
					
				def forward(self, x):
					x = self.layer(x)
					return x
			net = Net()
			print(net)
			```
- 超参数设置
	- 简单的设置
	- 代码
		- ```python
			# 设置超参数
			import torch.optim as optim
			import torch.nn as nn
			optimizer = optim.Adam(net.parameters(), lr=0.02)
			# 使用交叉熵损失计算
			loss_func = nn.CrossEntropyLoss()
			EPOCH = 500
			```
- 训练及可视化训练过程
	- 使用matplotlib的交互模式进行数据刷新。
	- 代码
		- ```python
			# 训练
			import matplotlib.pyplot as plt
			%matplotlib qt5
			plt.ion()
			plt.show()
			
			for epoch in range(EPOCH):
				pred = net(x)
				loss = loss_func(pred, y)
				
				# 固定三步
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				# 绘制过程
				if epoch % 1 == 0:
					# 我的机器训练太快，，，所以这里暂停一下方便可视化输出
					import time
					time.sleep(1)
					plt.cla()
					# 这里注意算误差和此处不一样，只有经过一个softmax之后概率最大的才是预测值
					pred = torch.max(F.softmax(pred), 1)[1]
					pred_y = pred.data.numpy().squeeze()
					target_y = y.data.numpy()
					plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='PiYG')
					accuracy = sum(pred_y == target_y)/len(y)  # 计算与分类准确的数目
					plt.text(3, -4, 'Accuracy={:.2f}'.format(accuracy), fontdict={'size': 10, 'color':  'red'})
					plt.pause(0.1)
			plt.ioff()
			plt.show()
			```
	- 训练过程
		- ![](/asset/2019-04-18/rst.gif)


## 补充说明
- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
- 具体代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Pytorch/ClassificationDemo)，欢迎star或者fork。（开发环境为Jupyter）
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。