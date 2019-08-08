---
layout:     post
title:      PyTorch-自编码器实现
subtitle:   利用自编码器降维数据
date:       2019-04-24
author:     zhouchen
header-img: img/post-bg-pytorch.png
catalog: true
tags:
    - PyTorch
---
---
title:      PyTorch构建自编码器
date:       2019-04-24
categories: 'PyTorch'
description: 演示PyTorch中如何构建GAN网络。
tags:
    - 深度学习
    - PyTorch
updated: 
music-id: 
---

# 自编码器AutoEncoder
- 几乎所有神经网络的入门书籍都会提到自编码器，其实自编码器是一种典型的非监督学习的神经网络，它在数据核心特征提取方面效用巨大。
- 步骤
	- 获取数据集（对Mnist数据集进行自编码）
		- 代码
			- ```python
				import torch
				import torchvision.datasets as datasets
				import torchvision.transforms as transforms
				from torch.utils.data import DataLoader
				
				torch.manual_seed(2019)
				
				train_data = datasets.MNIST(
				    root='data/',
				    train=True,
				    transform=transforms.ToTensor(),
				    download=True,  # 第一次使用需要下载
				)
				
				dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)  # 为了配合GPU性能最好使用2^n
				```
	- 搭建网络结构
		- 编码解码过程是相反的
		- 其实编码器的原理可以理解为对主要特征的抽象
		- 代码
			- ```python
				# 模型构建
				import torch.nn as nn
				class AutoEncoder(nn.Module):
				    def __init__(self):
				        super(AutoEncoder, self).__init__()
				        # 编码
				        self.encoder = nn.Sequential(
				            nn.Linear(28*28, 128),
				            nn.Tanh(),
				            nn.Linear(128, 64),
				            nn.Tanh(),
				            nn.Linear(64, 32),
				            nn.Tanh(),
				            nn.Linear(32, 16),
				            nn.Tanh(),
				            nn.Linear(16, 3),  # 利用全连接压缩为3个特征
				        )
				        # 解码
				        self.decoder = nn.Sequential(
				            nn.Linear(3, 16),
				            nn.Tanh(),
				            nn.Linear(16, 32),
				            nn.Tanh(),
				            nn.Linear(32, 64),
				            nn.Tanh(),
				            nn.Linear(64, 128),
				            nn.Tanh(),
				            nn.Linear(128, 28*28),
				            nn.Sigmoid(),  # 利用Sigmoid将输出映射到(0,1)
				        )
				
				    def forward(self, x):
				        encoded = self.encoder(x)
				        decoded = self.decoder(encoded)
				        return encoded, decoded
				
				autoencoder = AutoEncoder()
				print(autoencoder)
				```
	- 训练网络
		- 时间关系，这里只训练了20轮
		- 代码
			- ```python
				import torch.optim as optim
				optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
				loss_func = nn.MSELoss()
				
				for epoch in range(20):
				    for step, (b_x, b_y) in enumerate(dataloader):
				        x_raw = b_x.view(-1, 28*28)
				        x_true = b_x.view(-1, 28*28)
				
				        encoded, decoded = autoencoder(x_raw)
				
				        loss = loss_func(decoded, x_true) 
				        
				        optimizer.zero_grad()
				        loss.backward() 
				        optimizer.step()
				    print("Epoch:{} Loss:{}".format(epoch, loss.data))
				```
	- 利用AutoEncoder我们可以将原图片压缩（经过encode和decode）
		- 代码
			- ```python
				# 编码后解码效果
				import numpy as np
				import matplotlib.pyplot as plt
				test_data = train_data.data[:5]
				
				for i in range(5):
				    
				    plt.subplot(2, 5, i+1)
				    plt.imshow(test_data[i].numpy(), cmap='gray')
				    plt.title('true')
				    
				    plt.subplot(2, 5, i+6)
				    _, end_data = autoencoder(test_data[i].view(-1, 28*28).type(torch.FloatTensor))
				    plt.imshow(np.reshape(end_data.data.numpy(), (28, 28)), cmap='gray')
				    plt.title('encode')
				plt.show()
				```
		- 运行结果
			- 可以看到，主要的特征都保留了
			- ![](https://img-blog.csdnimg.cn/20190424185723957.png)
	- 利用AutoEncoder进行特征区分
		- 在上面的例子中，利用编码器的编码解码达到了特征抽取的效果（这类似主成分分析，有时候效果更好一些）
		- 其实，只利用编码器，我们可以将数据大致区分开
		- 代码
			- ```python
				# 数据区分
				from mpl_toolkits.mplot3d import Axes3D
				from matplotlib import cm
				test_data = train_data.data[:100].view(-1, 28*28).type(torch.FloatTensor)/255.
				end_data, _ = autoencoder(test_data)
				
				fig = plt.figure(2)
				ax = Axes3D(fig)
				# 数据特征被压缩到了三个值
				
				x = end_data.data[:, 0].numpy()
				y = end_data.data[:, 1].numpy()
				z = end_data.data[:, 2].numpy()
				# 为了区分划分是否合适， 将标签拿出来
				target = train_data.targets[:100].numpy()
				for i in range(100):
				    x_data, y_data, z_data, v_data = x[i], y[i], z[i], target[i]
				    c = cm.viridis(int(255*target[i] / 8))
				    ax.text(x[i], y[i], z[i], target[i], backgroundcolor=c)
				    
				ax.set_xlim(x.min(), x.max())
				ax.set_ylim(y.min(), y.max())
				ax.set_zlim(z.min(), z.max())
				plt.show()
				```
		- 运行结果
			- 使用cmap映射不同标签的颜色得到一个3D图，可以看到数据还是被大致区分开的（这就是一种典型的无监督学习聚类）
			- ![](https://img-blog.csdnimg.cn/20190424191255119.png)
- 补充说明
	- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体完整代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）