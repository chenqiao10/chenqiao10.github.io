---
layout:     post
title:      PyTorch-搭建GAN
subtitle:   利用GAN生成完美二次曲线
date:       2019-04-25
author:     zhouchen
header-img: img/post-bg-pytorch.png
catalog: true
tags:
    - PyTorch
---

# 简单搭建GAN
- 本案例通过一个函数区间让网络进行学习，最终稳定在一个函数区间内。
- 步骤
	- 优秀作品生成
		- 代码
			- ```python
				# 产生32个优秀作品交给discriminator鉴别
				temp = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
				good_paintings = torch.from_numpy(temp * np.power(POINTS, 2) + (temp-1)).float()
				```
	- 模型构建
		- 代码
			- ```python
				# 搭建网络结构
				# Generator，主要任务是产生一些随机作品
				G = nn.Sequential( 
				    nn.Linear(IDEA_NUM, 128),
				    nn.ReLU(),
				    nn.Linear(128, MAX_POINTS),  # 利用全连接提取随机特征
				)
				# Discriminator，主要任务是鉴别作品
				D = nn.Sequential( 
				    nn.Linear(MAX_POINTS, 128),
				    nn.ReLU(),
				    nn.Linear(128, 1),
				    nn.Sigmoid(),  # 映射为概率
				)
				
				opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
				opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
				
				```
	- 训练过程
		- 代码
			- ```python
				# 下面是训练的交互过程，使用matplotlib交互模式
				%matplotlib qt5
				plt.figure(figsize=(12, 8))
				plt.ion()
				
				for epoch in range(10000):
				    
				    g_ideas = torch.randn(BATCH_SIZE, IDEA_NUM)  # 随机产生idea
				    g_paintings = G(g_ideas)  # Generator生成作品
				    
				    # 接着拿优秀作品与G产生的作品给D，让D判断这两批作品是优秀作品的概率
				
				    prob0 = D(good_paintings)
				    prob1 = D(g_paintings)
				    
				    # 计算有多少来自优秀作者的作品猜对了，有多少来自G的画猜对了，最大化猜对的次数
				    # log(D(x)) + log(1-D(G(z))这就是论文提到的，但是torch没有最大化score只有最小化loss，这是一致的
				    D_loss = - torch.mean(torch.log(prob0) + torch.log(1. - prob1))
				    G_loss = torch.mean(torch.log(1. - prob1))
				    
				    # 网络调整
				
				    opt_D.zero_grad()
				    D_loss.backward(retain_graph=True)
				    opt_D.step()
				
				    opt_G.zero_grad()
				    G_loss.backward()
				    opt_G.step()
				
				    if epoch % 30 == 0:
				        plt.cla()
				        plt.plot(POINTS[0], g_paintings.data.numpy()[0], c='green', lw=3, label='Generated painting',)
				        # 始终显示上下边界
				        plt.plot(POINTS[0], 2 * np.power(POINTS[0], 2) + 1, c='blue', lw=3, label='above bound')
				        plt.plot(POINTS[0], 1 * np.power(POINTS[0], 2) - 1, c='red', lw=3, label='below bound')
				        plt.text(-0.6, 2.7, 'D accuracy={:.2f}'.format(prob0.data.numpy().mean()), fontdict={'size': 13})
				        plt.text(-0.6, 2.5, 'D score={:.2f}'.format(-D_loss.data.numpy()), fontdict={'size': 13})
				        plt.ylim((-3, 3))
				        plt.legend(loc='upper center', fontsize=10)
				        plt.draw()
				        plt.pause(0.01)
				
				plt.ioff()
				plt.show()
				```
		- 运行结果
			- 可以看到，只要不断训练GAN最终产生符合高分的优秀作品
			- ![](https://img-blog.csdnimg.cn/20190425132159414.gif)
- 补充说明
	- 本案例使用PyTorch框架，如果你是神经网络的新手建议你使用这个框架，上手容易，网络搭建结构化。（参考莫烦教程）
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体完整代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）