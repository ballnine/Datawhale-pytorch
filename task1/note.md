# 线性回归

`random.shuffle()`

方法将序列的所有元素随机排序。

`view()`

返回一个有相同数据但大小不同的tensor。 返回的tensor必须有与原tensor相同的数据和相同数目的元素，但可以有不同的大小。一个tensor必须是连续的contiguous()才能被查看。

例：
```
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])
```

# softmax
离散数据预测

# 多层感知机
+ relu

  ReLU函数只保留正数元素，并将负数元素清零

+ tanh

  tanh（双曲正切）函数可以将元素的值变换到-1和1之间

+ sigmod

  sigmoid函数可以将元素的值变换到0和1之间