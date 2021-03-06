---
title: 一行Python可以实现什么
date: '2018-05-01'
categories: 'Python'
description: 一行Python代码可以实现什么丧心病狂的功能。
tags: 
    - Python
updated: 2018-05-01
music-id: 1297532485
---


## 简介
一行Python代码可以实现什么丧心病狂的功能。


## 输出型

### 格式化输出乘法口诀表
- 代码
```python
print('\n'.join([' '.join(["{:2d} *{:2d} = {:2d}".format(j, i, i*j) for j in range(1, i+1)]) for i in range(1, 10)]))
```
- 结果
![](/asset/2018-05-01/mul.png)

### 打印迷宫
- 代码
```python
print(''.join(__import__('random').choice('\u2571\u2572') for i in range(50*24)))
```
- 结果
![](/asset/2018-05-01/maze.png)

### 打印爱心图案
- 代码
```python
print('\n'.join([''.join([('love'[(x-y) % len('love')] if ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <=0 else ' ')for x in range(-30, 30)]) for y in range(15, -15, -1)]))
```
- 结果
![](/asset/2018-05-01/love.png)

### 字符串高频字符
- 代码
```python
print(max("AASFCscwe", key="AASFCscwe".count))
```


## 服务型

### 开启Web文件服务
- 代码
```python
!python -m http.server 8088 
```
- 结果
![](/asset/2018-05-01/web.png)


## 运算型

### 变量互换
- 代码
```python
a, b = b, a
```

### 多维数组Flatten
- 代码
```python
flatten = lambda x : [y for l in x for y in flatten(l)] if isinstance(x, list) else [x]
```
- 结果
![](/asset/2018-05-01/flatten.png)


## 补充说明
具体项目代码上传至[我的Github](https://github.com/luanshiyinyang/PythonScripts/tree/OneLineCode)，欢迎Star或者Fork。博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。