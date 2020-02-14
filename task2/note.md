+ 文本预处理

  文本是一类序列数据，一篇文章可以看作是字符或单词的序列，本节将介绍文本数据的常见预处理步骤，预处理通常包括四个步骤：

  + 读入文本
  + 分词
  + 建立字典，将每个词映射到一个唯一的索引（index）
  + 将文本从词的序列转换为索引的序列，方便输入模型

+ 语言模型

  马尔可夫假设

+ RNN

  `result.scatter(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1`

  将src中的所有值按照index确定的索引写入本tensor中。其中索引是根据给定的dimension，dim按照gather()描述的规则来确定。

  注意，index的值必须是在_0_到_(self.size(dim)-1)_之间，

  参数： - input (Tensor)-源tensor - dim (int)-索引的轴向 - index (LongTensor)-散射元素的索引指数 - src (Tensor or float)-散射的源元素

  例如：
  ```
  >>> x = torch.rand(2, 5)
  >>> x

  0.4319  0.6500  0.4080  0.8760  0.2355
  0.2609  0.4711  0.8486  0.8573  0.1029
  [torch.FloatTensor of size 2x5]

  >>> torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

  0.4319  0.4711  0.8486  0.8760  0.2355
  0.0000  0.6500  0.0000  0.8573  0.0000
  0.2609  0.0000  0.4080  0.0000  0.1029
  [torch.FloatTensor of size 3x5]

  >>> z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)
  >>> z

  0.0000  0.0000  1.2300  0.0000
  0.0000  0.0000  0.0000  1.2300
  [torch.FloatTensor of size 2x4]
  ```