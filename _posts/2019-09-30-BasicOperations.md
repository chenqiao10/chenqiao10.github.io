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


## Tensor索引和切片
- C语言风格
  - 通过多层下标进行索引。![](/asset/2019-09-30/c_index.png)
- numpy风格
  - 通过多层下标索引，写在一个中括号内，使用逗号分隔。![](/asset/2019-09-30/np_index.png)
- Python风格
  - `array[start:end:step, start:end:step, ...]`可以缺省，start和end缺省时取从开端到结尾。同时，默认从第一个维度开始取，几个冒号则从开始取几个维度，后面的剩余维度全取。同样，上述省略号表示后面的维度都取，等同于不写的含义（但是，当省略号出现在中间则不能不写）。![](/asset/2019-09-30/slice.png)
- selective index
  - `tf.gather(a, axis, indices)`
  - axis表示指定的收集维度，indices表示该维度上收集那些序号。
  - `tf.gather_nd(a, indices)`
  - indices可以是多维的，按照指定维度索引。
  - `tf.boolean_mask(a, mask, axis)`
  - 按照布尔型的mask，对为True的对应取索引（支持多层维度）。
  - 演示。![](/asset/2019-09-30/selective.png)


## Tensor维度变换
- `tf.reshape(a, shape)`
  - 将Tensor调整为新的合法shape，不会改变数据，只是改变数据的理解方式。（reshape中维度指定为-1表示自动推导，类似numpy）![](/asset/2019-09-30/reshape.png)
- `tf.transpose(a, perm)`
  - 将原来Tensor按照perm指定的维度顺序进行转置。![](/asset/2019-09-30/transpose.png)
- `tf.expand_dims(a, axis)`
  - 在指定维度的前面（axis为正数）或者后面（axis为负数）增加一个新的空维度。![](/asset/2019-09-30/expand.png)
- `tf.squeeze(a, axis)`
  - 消去指定的可以去掉的维度（该维度值为1）。![](/asset/2019-09-30/squeeze.png)


