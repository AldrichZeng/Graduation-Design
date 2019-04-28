### 基于剪枝的神经网络压缩与加速
###### 曾曜 - 本科毕业设计 指导老师：李文中（南京大学）、陆伟（西北工业大学）

# 权重剪枝（sparsify）

#### LeNet-5 on MNIST-10（卷积层）
|剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|batch_size|
|:-----|:------|:-------|:------|:-----|:--------|:---------|
|0%    |/      |0.9902  |185    | 0.001|无       |256|
|50%   |0.9883 |0.9898  |154    |0.001 |无       |256|
|90%   |       |        |       |      |         ||

#### AlextNet on CIFAR-10（卷积层）
|层  |剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|batch_size|备注|
|:---|:-----|:------|:-------|:------|:-----|:--------|:---------|:---|
|Conv|0%    |/      |0.75474 |636    | 0.001|0.1/100epoch|128||
|Conv|50%   |0.4386 |0.6071  |607    |0.001 |0.1/100epoch|128||
|Conv|90%   |0.4579 |0.6284  |174    |0.001 |0.1/100epoch|128|保留Conv1|
|FC  |50%   |0.6873|0.7245   |283    |0.001 |0.1/100epoch|128||


#### VGG-16 on CIFAR-10
|剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|batch_size|
|:-----|:------|:-------|:------|:-----|:--------|:---------|
|0%    |/      |0.6552  |网上下载|      |         |16|
|50%   |       |  |     |       |      |         |16|


————2019年4月28日
# 卷积核剪枝
#### LeNet-5 on MNIST-10

#### AlextNet on CIFAR-10

#### VGG-16 on CIFAR-10

#### 参考论文
1. 《Deep compression - Compressing Deep Neural Networks With Pruning, Tranied Quantization And Huffman Coding》
2. 《Leaning both Weights and Connections for Efficient Neural Networks》
3. 《Pruning Filters For Efficient ConvNets》

# CP分解
对于未剪枝的LeNet-5的模型，采用CP分解后，首次测试准确率为0.11，经过多次迭代后可以达到90%以上。

对于剪枝率90%的LeNet-5模型，采用CP分解后，首次测试准确率为0.8732，经过一次迭代后，就可以达到Accuracy : 0.9788

学习率是否要很小？学习率依旧选取0.001，如果太小会难以收敛，训练效果并不好。
#### LeNet-5 on MNIST-10
|剪枝率|首次测试|accuracy|迭代次数|学习率|学习率衰减|batch_size|
|:-----|:------|:-------|:------|:-----|:--------|:---------|
|0%    |0.11   |>0.90||0.001||256|
|90%|0.8732|>0.9788||0.001||256|

————2019年4月9日
# Tucker分解

#### 参考论文
4. 《Speeding-up Convolutional Neural Networks Using Fine-tuned CP-decomposition》
5. 《Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications》
