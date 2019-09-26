---
title: Jupyter进阶教程
date: '2019-08-28'
categories: '杂项'
description: 介绍Jupyter的一些高级操作。
tags: 
    - Jupyter
updated: 
music-id: 
---
## 简介
- Jupyter Notebook是一款交互式的笔记本或者代码编辑器，支持几十种编程语言，不过主要还是为以Python和R为首的数据科学使用，其前身是Ipython，关于Jupyter的基本安装和使用不多提及。
- 本教程主要提及一些Jupyter的进阶技巧，基本的使用及快捷键等在入门教程中已经提及了。
- 主要介绍常用的shell命令及Jupyter封装的魔法方法，探索外部插件的使用。

## Shell命令
- 在Jupyter笔记本中，任何操作系统支持的shell命令都可以通过`!`跟命令行输入的命令执行。
  - 如![](/asset/2019-08-28/shell.png)
- 同时，shell命令可以通过`$`+变量名来获得出现过的Python变量的值。
  - 如![](/asset/2019-08-28/shell-var.png)

## 魔法命令（Magics）
- 说明
  - 魔法命令是基于Jupyter内核所设计的一些很方便的命令，类似Linux命令，它都是由`%`+命令名或者`%%`+命令名组成，其本质上都是使用Python实现的。其中，最常见的是`!cd`操作在Jupyter中是无效的，使用`%cd`则可以切换Jupyter的工作目录。**使用`%lsmagic`可以列出所有魔法命令，如图。访问[官网](https://ipython.readthedocs.io/en/stable/interactive/magics.html)可以查看所有魔法命令说明。![](/asset/2019-08-28/magics.png)**
- 分类
  - 行魔法命令（line magics）
    - 作用于单行范围，`%`开头。
  - 单元魔法命令（cell magics）
    - 作用于多行甚至整个cell，`%%`开头。
- 常用魔法命令
  - `autosave xx`
    - 设置自动保存间隔的秒数。
    - ![](/asset/2019-08-28/autosave.png)
  - `matplotlib inline`
    - 设置matplotlib的图表在notebook的一个单元内显示，通常导入matplotlib前就需要设置。
    - ![](/asset/2019-08-28/matplotlib.png)
  - `%time`
    - 输出单元格代码执行时间，`%timeit`类似，不过会多次运行求平均时间，多次运行的此时可以通过`-n`选项指定。
    - ![](/asset/2019-08-28/time.png)
  - `%%language`
    - 将整个单元格作为指定的编程语言执行，如HTML，Markdown，JS。
    - ![](/asset/2019-08-28/language.png)


## 外部插件
- 说明
  - 作为一个开源工具，很多开发者为Jupyter开发了很多有效的插件如数据库连接、代码拼写检查、代码折叠（很好用）等。
- 安装
  - 方法
    - 在指定环境下使用pip安装插件，启用插件，重启jupyter notebook。
  - 代码折叠
    - ```shell
        pip install jupyter_contrib_nbextensions
        jupyter contrib nbextension install --user
        jupyter nbextension enable spellchecker/main
        jupyter nbextension enable codefolding/main
        ```
    - 安装后可以进行折叠等功能。![](/asset/2019-08-28/extension.png)
  - 幻灯片展示
    - ```shell
        pip install RISE
        jupyter-nbextension install rise --py --sys-prefix
        jupyter-nbextension enable rise --py --sys-prefix 
        ```
    - 安装后可以进行幻灯片展示代码等。![](/asset/2019-08-28/ppt.png)


## 宏命令（Macros）
- 说明
  - 往往有一些重复的代码，每次创建notebook都要重新输入，如包的导入，其实可以保存为宏命令，在任何notebook中使用。
- 创建macro
  - `%macro -q __hello 4`
    - 创建一个名为__hello的宏，使用`-q`选项指定宏名称，后面紧跟需要作为宏的已执行单元格编号。
  - `%store __hello`
    - 保存该宏命令。
- 使用macro
  - `%store -r __hello_world`
    - 载入保存过的宏命令，使用`-r`紧跟宏命令名称。
  - `macro_name`
    - 单元格内执行宏命令。
- ![](/asset/2019-08-28/macro.png)


## 外部代码
- `%load file.py`
  - 导入并执行Python脚本文件，执行后会将脚本文件的内容copy到单元格内执行，并自动注释load命令。
  - 如创建一个py文件只有`a = 100`这一行代码，执行如图。![](/asset/2019-08-28/load.png)
- `%run file.py`
  - 执行py文件，相当于在cell中copy并执行，但是不现实copy到的内容。（不同于`!python file.py`，run命令会获得py文件的所有变量值）
  - 可以添加很多选项进行运行，详情查看官网。
  - ![](/asset/2019-08-28/run.png)


## 补充说明
- 参考网络上一些教程，做了一些改动和删减。这里提一下，如果既想要jupyter notebook的轻便和IDE的丰富功能，那么jupyter lab将是很好的额选择，其界面是这样的。![](/asset/2019-08-28/jupyterlab.png)
- 博客已经同步至我的[个人博客网站](https://luanshiyinyang.github.io)，欢迎访问查看最新文章。
- 如有错误或者疏漏之处，欢迎指正。