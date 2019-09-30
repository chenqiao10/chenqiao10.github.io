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


