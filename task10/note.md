+ 图像分类实例2

  图像增强：

  训练集：随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和宽均为224像素的新图像；
  以0.5的概率随机水平翻转；
  随机更改亮度、对比度和饱和度；
  对各个通道做标准化

  测试集：只做确定的操作
  resize;
  将图像中央的高和宽均为224的正方形区域裁剪出来;

+ gan

  分类模型（判别模型），学习x->y，p(y|x)

  生成模型（生成学习）,没有标签，生成与源数据类似却不同的数据，p(x)

  使用判别模型帮助生成式学习（RNN）

  gan——生成神经网络、分类神经网络（二分类）

  loss:分类二分类最小化，生成二分类最大（y为0）

+ DCGAN

  将多层卷积层运用到生成器和判别器中来模拟图像

  生成网络，3个g_block,一个转置卷积层

  判别器：leaky relu