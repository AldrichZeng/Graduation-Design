# 基于剪枝的神经网络压缩与加速
###### 曾曜 - 本科毕业设计

### 剪枝（稀疏化）

#### LeNet-5 on MNIST-10
|剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|
|:-----|:------|:-------|:------|:-----|:--------|
|0%    |/      |0.9902  |185    | 0.001|无       |
|50%   |0.9883 |0.9898  |154    |0.001 |无       |
|90%   |      |      |   |     |

#### AlextNet on CIFAR-10
|剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|
|:-----|:------|:-------|:------|:-----|:--------|
|0%    |/      |0.75474 |636    | 0.001|0.1/100epoch|
|50%   |0.4386 |0.6071  |607    |0.001 |0.1/100epoch|
|90%   |      |      |   |     |

#### VGG-16 on CIFAR-10
|剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|
|:-----|:------|:-------|:------|:-----|:--------|
|0%    |/      |0.6552  |网上下载|      |         |
|50%   |       |  ||      |         |


————2019年4月28日

#### 参考论文
1. 《Deep compression - Compressing Deep Neural Networks With Pruning, Tranied Quantization And Huffman Coding》
2. 《Leaning both Weights and Connections for Efficient Neural Networks》
3. 《Pruning Filters For Efficient ConvNets》

### 低秩分解（张量分解/矩阵分解）
对于未剪枝的LeNet-5的模型，采用CP分解后，首次测试准确率为0.11，经过多次迭代后可以达到90%以上。

对于剪枝率90%的LeNet-5模型，采用CP分解后，首次测试准确率为0.8732，经过一次迭代后，就可以达到Accuracy : 0.9788

学习率是否要很小？学习率依旧选取0.001，如果太小会难以收敛，训练效果并不好。

————2019年4月9日

#### 参考论文
4. 《Speeding-up Convolutional Neural Networks Using Fine-tuned CP-decomposition》
5. 《Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications》
