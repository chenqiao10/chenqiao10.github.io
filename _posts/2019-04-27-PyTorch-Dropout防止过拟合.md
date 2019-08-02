---
layout:     post
title:      PyTorch-Dropout防止过拟合
subtitle:   演示利用Dropout防止过拟合
date:       2019-04-27
author:     zhouchen
header-img: img/post-bg-pytorch.png
catalog: true
tags:
    - PyTorch
---

# 使用Dropout缓解过拟合
- 本案例将演示在PyTorch中如何使用Dropout缓解过拟合。
- 介绍
	- 过拟合指的是模型随着训练在训练集上的损失不断降低，但是在某个时间点之后再测试集上的损失却开始飙升，这是因为随着训练模型对训练数据的拟合效果越来越好，从而使得泛化能力降低，这个在深度学习中一般称为过拟合问题。
	- 过拟合有很多解决办法，如添加惩罚项（L1，L2正则化）、EarlyStopping技术等，在神经网络中有一种独有的方式就是Dropout，简单的说就是随机丢弃上一层与当前层的部分神经元连接从而控制拟合过程。
	- 大量实验表明在神经网络中0.5的Dropout有较好的效果。
- 步骤
	- 生成数据集
		- 生成类似分布数据集，可视化数据分布。
		- 这样的数据集很容易过拟合。
		- 代码
			- ```python
				import torch
				import matplotlib.pyplot as plt
				%matplotlib inline
				torch.manual_seed(2019)
				
				sample_num = 30  # 样本数据量
				# 训练数据
				x_train = torch.unsqueeze(torch.linspace(-1, 1, sample_num), dim=1)
				y_train = x_train + 0.3 * torch.normal(torch.zeros(sample_num, 1), torch.ones(sample_num, 1))  # 指定均值、标准差的正太数据
				
				# 验证数据
				x_valid = torch.unsqueeze(torch.linspace(-1, 1, sample_num), dim=1)
				y_valid = x_valid + 0.3 * torch.normal(torch.zeros(sample_num, 1), torch.ones(sample_num, 1))
				
				# 可视化数据分布
				plt.figure(figsize=(12, 8))
				plt.scatter(x_train.data.numpy(), y_train.data.numpy(), c='red', label='train')
				plt.scatter(x_valid.data.numpy(), y_valid.data.numpy(), c='blue', label='valid')
				plt.legend(loc='best')
				plt.ylim((-2.5, 2.5))
				plt.show()
				```
		- 数据分布
			- ![](https://img-blog.csdnimg.cn/20190427103100157.png)
	- 模型构建
		- 搭建一个没有添加Dropout和一个添加Dropout的模型
		- 代码
			- ```python
				import torch.nn
				hidden_num = 100  # 隐层神经元越多越容易过拟合
				net_overfitting = torch.nn.Sequential(
				    torch.nn.Linear(1, hidden_num),
				    torch.nn.ReLU(),
				    torch.nn.Linear(hidden_num, hidden_num),
				    torch.nn.ReLU(),
				    torch.nn.Linear(hidden_num, 1),
				)
				
				net_dropout = torch.nn.Sequential(
				    torch.nn.Linear(1, hidden_num),
				    torch.nn.Dropout(0.5), 
				    torch.nn.ReLU(),
				    torch.nn.Linear(hidden_num, hidden_num),
				    torch.nn.Dropout(0.5),
				    torch.nn.ReLU(),
				    torch.nn.Linear(hidden_num, 1),
				)
				```
	- 进行训练
		- 代码
			- ```python
				import torch.optim
				import numpy as np
				optimizer_overfit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
				optimizer_dropout = torch.optim.Adam(net_dropout.parameters(), lr=0.01)
				loss_func = torch.nn.MSELoss()  # 这个回归预测问题mse具有较好效果
				
				losses_train_1, losses_train_2, losses_valid_1, losses_valid_2 = [], [], [], []
				# 开始训练
				for t in range(500):
				    pred_overfit = net_overfitting(x_train)
				    pred_dropout = net_dropout(x_train)
				    loss_overfit = loss_func(pred_overfit, y_train)
				    loss_dropout = loss_func(pred_dropout, y_train)
				
				    optimizer_overfit.zero_grad()
				    optimizer_dropout.zero_grad()
				    loss_overfit.backward()
				    loss_dropout.backward()
				    optimizer_overfit.step()
				    optimizer_dropout.step()
				    
				    if t % 10 == 0:
				        # 改成eval模式即测试模式
				        net_overfitting.eval()
				        net_dropout.eval() 
				        # 过拟合网络
				        losses_train_1.append(loss_overfit.data.numpy())  # 将训练集损失加入列表
				        pred1 = net_overfitting(x_valid)
				        losses_valid_1.append(loss_func(pred1, y_valid).data.numpy())
				        
				        # Dropout网络
				        losses_train_2.append(loss_dropout.data.numpy())
				        pred2 = net_dropout(x_valid)
				        losses_valid_2.append(loss_func(pred2, y_valid).data.numpy())
				        
				        # 切换回训练模式
				        net_overfitting.train()
				        net_dropout.train() 
				
				        
				plt.figure(figsize=(12, 8))
				plt.subplot(2, 1, 1)
				plt.plot(np.arange(len(losses_train_1)), losses_train_1, label='overfit train loss')
				plt.plot(np.arange(len(losses_train_1)), losses_valid_1, label='overfit valid loss')
				
				plt.subplot(2, 1, 2)
				plt.plot(np.arange(len(losses_train_1)), losses_train_2, label='dropout train loss')
				plt.plot(np.arange(len(losses_train_1)), losses_valid_2, label='dropout valid loss')
				plt.legend(loc='best')
				plt.show()
				```
		- 演示效果
			- 可以看到，在没有Dropout的网络上，训练集损失迅速降低，验证集损失反而有升高趋势。
			- 但是，添加了dropout，尽管训练效率变低，但是验证集上的效果被控制住，模型更有泛化能力。
			- ![](https://img-blog.csdnimg.cn/20190427105528966.png)
- 补充说明
	- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。（参考莫烦教程）
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体完整代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）