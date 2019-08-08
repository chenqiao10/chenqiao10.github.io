---
title:      PyTorch实现回归网络
date:       2019-04-19
categories: 'PyTorch'
description: 演示PyTorch中实现回归模型的方法。
tags:
    - 深度学习
    - PyTorch
updated: 
music-id: 
---
## 简介
- 本案例使用只含有一个隐层，20个神经元的简易神经网络进行二次回归，并可视化回归效果。


## 步骤
- 创建数据集
	- 为了方便演示，这里的特征只有一个维度，为x值，目标也只有一个维度，为y值，生成的数据近似为二次函数拟合函数。
	- 代码
		- ```python
			import torch
			import matplotlib.pyplot as plt
			# 创建-1到1之间均分的行向量x
			x = torch.linspace(-1, 1, 100)
			# x逆压缩为二维tensor
			x = torch.unsqueeze(x, dim=1)
			# 创建y目标分布
			y = x.pow(2) + 0.2*torch.rand(x.size())                 
			# 绘制散点图显示数据点分布
			plt.scatter(x.data.numpy(), y.data.numpy())
			plt.show()
			```
- 搭建网络结构
	- 由于只是一个简易回归，我们使用一个隐层参数去拟合数据即可。
	- 代码
		- ```python
				# 建立简单神经网络
				import torch
				import torch.nn.functional as F
				
				class Net(torch.nn.Module):
					def __init__(self, n_input, n_output):
						# 继承为规范写法
						super(Net, self).__init__()
						# 定义隐藏层，输入为输入数据维度，输出为隐层神经元数目，这里设定隐层神经元有20个
						self.hidden = torch.nn.Linear(n_input, 20)
						# 定义输出层，将隐藏层输出作为输入并线性输出（按照输出维度）
						self.output = torch.nn.Linear(20, n_output)
				
					def forward(self, x): 
						# 定义前向传播函数
						x = F.relu(self.hidden(x))  # 将线性输出激活为非线性
						x = self.output(x)  # 输出不需要激活
						return x
				
				# 按照定义结果创建一个网络
				net = Net(n_input=1, n_output=1)
				# 输出网络结构
				print(net)
				```
- 设定训练超参数
	- 代码
		- ```python
			# 定义优化函数，使用SGD作为优化函数
			optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
			# 定义损失函数，使用MSE作为损失函数
			loss_func = torch.nn.MSELoss()
			# 定义超参数
			epochs = 1000
			```
- 训练并可视化回归过程
	- 利用pyplot的交互模式进行过程显示
	- 代码
		- ```python
			# 进行训练并可视化此过程
			import matplotlib.pyplot as plt
			%matplotlib qt5
			
			plt.ion()
			plt.show()
			
			for epoch in range(epochs):
				pred = net(x)  # 进行预测
				loss = loss_func(pred, y)  # 计算损失
				optimizer.zero_grad()  # 清零上一轮对象保存的梯度
				loss.backward()  # 反向传播
				optimizer.step()  # 更新值传入模型参数
				
				if epoch % 5 == 0:
					# 每10轮更新一次图像拟合情况
					plt.cla()
					plt.scatter(x.data.numpy(), y.data.numpy())
					plt.plot(x.data.numpy(), pred.data.numpy(), lw=3, color='red')
					plt.text(0, 0, 'Loss={:.4f}'.format(loss.data.numpy()), fontdict={'color':'red'})
					plt.pause(0.1)
			print(net.paramenters())
			```
	- 拟合过程
		- ![](/asset/2019-04-19/rst.gif)
	- 训练结束时的神经元权重（不含偏置参数）
		- ![](/asset/2019-04-19/rst.png)


## 补充说明
- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
- 具体代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Pytorch/RegressionDemo)，欢迎star或者fork。（开发环境为Jupyter）
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。