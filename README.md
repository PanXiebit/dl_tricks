# dl_tricks
主要参考《深入理解 Tensorflow 架构设计与实现原理》这本书，实现 deep learning 中一些 tools 和 tricks.

## batch normalization
原论文笔记参考：[深度学习-Batch Normalization](https://panxiebit.github.io/2018/07/28/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-Batch-Normalization/#more)

在[这部分](https://github.com/PanXiebit/dl_tricks/blob/master/batch_normalization/Batch%20Normalization%20Tensorflow%E5%AE%9E%E7%8E%B0.ipynb)通过简单的三层神经网络，验证 bn 能保证激活值（也就是每一曾的线性输出）稳定在一个可靠的分布内，可以避免激活函数为 sigmoid 时，梯度陷入饱和区域的情况。这也允许我们在设置权重初始值时更随意一点，以及可以设置较大的学习率，从而加快训练速度。

在[py文件中](https://github.com/PanXiebit/dl_tricks/blob/master/batch_normalization/batch_norm.py) 是 batch normalization 的实现，需要注意的是在训练阶段和测试阶段，其计算方式是不一样的。  
- 在训练阶段，使用batch mean 和 batch variance 进行归一化，并通过可学习参数 beta 和 gamma 进行 shift 和 scale.  
- 在测试阶段，使用训练阶段统计的滑动平均值和方差来代替总体均值和方差，并对测试阶段的数据集进行归一化。

## model parameters
主要内容：  
- 模型参数的理解，存储节点，计算节点  
- 模型保存和恢复  
- 如何使用与训练模型，并进行 fine-tune

## input_pipeline
主要内容：
- tensorflow 流水线输入数据
