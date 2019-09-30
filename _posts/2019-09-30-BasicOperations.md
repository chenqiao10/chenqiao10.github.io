---
title: TensorFlow2基础操作
date: '2019-09-30'
categories: 'TensorFlow2'
description: 介绍TensorFlow2的一些基础API使用。
tags: 
    - TensorFlow2教程
updated: 
---
<img src='/asset/2019-09-21/tf2.gif' alt='' />


## 数据类型
- 说明
  - TensorFlow其实并没有那么神秘，为了适应自动求导和GPU运算，它应运而生。为了契合numpy的核心数据类型ndarray，其最核心的数据类型为Tensor，中文指张量（一般，数学上分标量，一维向量，二维矩阵，二维以上称为张量，当然在TF2中上述各种都是使用Tensor类型）。而Variable是对Tensor的一个封装，使其Tensor具有自动求导的能力（即可以被优化，这个类型是专为神经网络参数设定的）。

- Tensor
  - 数值类型
    - int, float, double
    - bool
    - string
    - 演示![](/asset/2019-09-30/datatype.png)
  - Variable
    - 创建及使用类似Tensor，只是多了trainable等属性。
    - 演示![](/asset/2019-09-30/variable.png)


## Tensor创建
- from numpy or list
  - TF的Tensor可以直接从numpy的矩阵或者符合矩阵规则的Pythonlist中生成。
  - 演示![](/asset/2019-09-30/from_np.png)
- 方法创建
  - `tf.zeros`
    - 接受参数为shape，创建全0的tensor。![](/asset/2019-09-30/zeros.png)
  - `tf.zeros_like`
    - 接受参数为tensor，创建根据该tensor的shape的全0的tensor。![](/asset/2019-09-30/zeros_like.png)
  - `tf.ones`
    - 类似tf.zeros
  - `tf.ones_like`
    - 类似tf.zeros_like
  - `tf.random.normal`
    - 接受参数为shape,mean,stddev，创建指定shape的tensor，数据从指定均值和标准差的正态分布中采样。![](/asset/2019-09-30/normal.png)
  - `tf.random.truncated_normal`
    - 接受参数同上，创建指定shape的tensor，数据从指定均值和标准差的正态分布截断后采样。
  - `tf.random.uniform`
    - 接受参数为shape,minval,maxval，创建指定shape的tensor，数据从指定最小值到最大值之间的均匀分布中生成。![](/asset/2019-09-30/uniform.png)
  - `tf.range`
    - 接受参数为limit，创建一维的start到limit的tensor。![](/asset/2019-09-30/range.png)
  - `tf.constant`
    - 类似tf.convert_to_tensor。


