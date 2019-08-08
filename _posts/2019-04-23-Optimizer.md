---
title:      PyTorch中不同的优化器函数
date:       2019-04-23
categories: 'PyTorch'
description: 演示PyTorch中使用不同的优化函数。
tags:
    - 深度学习
    - PyTorch
updated: 
music-id: 
---
## 简介
- 本案例简单使用未调参的几个不同优化器显示训练结果。


## 步骤
- 生成数据集
	- 使用之前的二次分布点数据集。
	- 代码
		- ```python
			x = torch.linspace(-1, 1, 1000)  # 维度为(1000)
			x = torch.unsqueeze(x, dim=1)  # 维度为(1000, 1)
			# 从给定序列生成正太数据
			y = x.pow(2) + 0.1 * torch.randn(x.size())
			
			plt.scatter(x.numpy(), y.numpy(), lw=0)
			plt.show()
			```
	- 可视化
		- ![](/asset/2019-04-23/data.png)
- 定义网络结构
	- 代码
		- ```python
			# 定义网络结构
			import torch.nn as nn
			import torch.optim as optim
			import torch.nn.functional as F
			class Net(nn.Module):
				def __init__(self):
					super(Net, self).__init__()
					self.hidden = torch.nn.Linear(1, 20)
					self.output = torch.nn.Linear(20, 1)
			
				def forward(self, x):
					x = F.relu(self.hidden(x))
					x = self.output(x) 
					return x
			```
- 构建网络集和优化器集合
	- 代码
		- ```python
			nets = [Net(), Net(), Net(), Net()]
			optimizers = [optim.SGD(nets[0].parameters(), lr=learning_rate),
							optim.Adam(nets[1].parameters(), lr=learning_rate), 
							optim.RMSprop(nets[2].parameters(), lr=learning_rate, alpha=0.9), 
							optim.Adagrad(nets[3].parameters(), lr=learning_rate), ]
			loss_func = nn.MSELoss()
			losses = [[], [], [], []]
			```
- 训练过程可视化
	- 代码
		- ```python
			for epoch in range(epochs):
				for step, (batch_x, batch_y) in enumerate(loader):
					for net, optimizer, loss_this in zip(nets, optimizers, losses):
						output = net(batch_x)
						loss = loss_func(output, batch_y) 
						optimizer.zero_grad() 
						loss.backward()
						optimizer.step() 
						loss_this.append(loss.data.numpy())
			```
	- 可视化
		- ![](/asset/2019-04-23/rst.png)


## 补充说明
- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
- 具体完整代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Pytorch/OptimizerDemo)，欢迎star或者fork。（开发环境为Jupyter）
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。