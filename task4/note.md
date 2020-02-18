+ 机器翻译及相关技术

  问题，输入输出长度不一致

  数据预处理：数据清洗，将乱码和不需要的去掉

  分词：转化成列表，`text.splie()`

  建立词典

  数据集每个batch长度固定，padding不足补长，计算损失时去掉，记录value_len

  + encoder-decoder（解决输入输出不等价）

  encoder:输入到隐藏状态（语义编码c）

  decoder:隐藏状态到输出

  + Sequence to Sequence模型

  decoder是一个RNN，h-1为隐藏状态

  encoder是一个RNN，每个单词通过embeding层输入转化为词向量作为输入，使用lstm，深度循环神经网络，输出state包括记忆细胞和隐层状态

  + 问题：只考虑局部最优解

    bean search：每步考虑结果最好的beam个继续循环

+ 注意力机制与Seq2seq模型

  seq2seq可能导致梯度消失，句子越长，效果下降；目标词语可能只与原输入部分词语有关。

  注意力机制显著建模选择过程

  softmax屏蔽操作，排除padding信息

  + dot product attention（点击注意力层)

  + multilayer perceptron attention

  修改decoder，添加一个MLP注意层

+ Transformer 

  整合CNN和RNN的优势