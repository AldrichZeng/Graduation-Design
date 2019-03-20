import torch
import numpy as np
import vgg

# 针对VGG网络的剪枝
def replace_layers(module,old_mod,new_mod):
    """
    在Module中在找到对应于old_mod[i]的Conv层或Batch Norm层，并替换
    :param module: torch.nn.Module类型，ResNet或VGG模型
    :param old_mod: 以torch.nn.Module为元素的列表，旧的层结构
    :param new_mod: 以torch.nn.Module为元素的列表，新的层结构
    :return:
    """
    for i in range(len(old_mod)):
        if module is old_mod[i]: # 若来自同一个对象，如：都是Conv2d或都是Batch Norm
            return new_mod[i]   # 返回第i个层结构
    return module


def prune_conv_layer(model, layer_index, filter_index):
    """
    :param model: VGG或ResNet模型
    :param layer_index: 整数，表示第几个卷积层（从1到13）
    :param filter_index: 整数列表，要删layer_index层中的哪个filter
    :return:
    """
    conv=None                          #获取要删filter的那层conv
    batch_norm=None                    #如果有的话：获取要删的conv后的batch normalization层
    next_conv=None                     #如果有的话：获取要删的那层后一层的conv，用于删除对应通道
    i=0

    '''
    在model.featues中查找第layer_index个的Conv层（从0开始），并找到后继的Conv层和Batch Norm层
    '''
    for mod in model.features:
        if conv is not None:
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                next_conv = mod  # 要进行剪枝的Conv的后继Conv层
                break
            elif isinstance(mod,torch.nn.modules.BatchNorm2d):
                batch_norm=mod  # 要进行剪枝的Conv的后继Batch Norm层
            else:
                continue
        if isinstance(mod,torch.nn.modules.conv.Conv2d):  # 若是卷积层
            i+=1
            if i==layer_index:
                conv=mod   # 保存要剪枝的Conv层

    if conv is not None:
        new_conv = torch.nn.Conv2d( in_channels=conv.in_channels,  # 创建新的conv替代要剪枝的Conv层
                                    out_channels=conv.out_channels - len(filter_index),  # 改变输出通道数，即减少卷积核数量
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    dilation=conv.dilation,
                                    groups=conv.groups,
                                    bias=(conv.bias is not None))

        '''
        创建new_weights和new_bias,
        在model.features中找到mod，然后在mod中找到对应的Conv层（可能要剪枝多个Conv层），进行替换，生成新的features
        '''

        # 复制weights
        old_weights = conv.weight.data.cpu().numpy()  # 获取卷积层的weight
        new_weights = new_conv.weight.data.cpu().numpy()  # 获取卷积层的weight
        # 对于不在filter_index中的行，复制到new_weights中（这里存在一个问题，如果Python广播机制中，不同维度的向量和矩阵有时候无法匹配）
        new_weights[:]=old_weights[[i for i in range(old_weights.shape[0])
                                    if i not in filter_index]]  #复制剩余的filters的weight

        # 复制bias
        if conv.bias is not None:
            old_bias = conv.bias.data.cpu().numpy()  # 将Tensor转化为numpy
            new_bias = new_conv.bias.data.cpu().numpy()
            new_bias[:] = old_bias[[i for i in range(old_bias.shape[0]) if i not in filter_index]]  # 复制剩余的filters的bias

        if torch.cuda.is_available():
            new_conv.cuda()

        # 在model.features中找到mod，然后在mod中找到对应的Conv层，进行替换，生成新的features
        model.features = torch.nn.Sequential(
            *(replace_layers(mod, [conv], [new_conv]) for mod in model.features))

    '''
    删除后继的Bath Norm的对应连接
    '''
    if batch_norm is not None: # 后继的Bath Norm不为空
        # 创建new_batch_norm
        new_batch_norm=torch.nn.BatchNorm2d(new_conv.out_channels)

        # 删除多余的weight
        old_weights = batch_norm.weight.data.cpu().numpy()
        new_weights = new_batch_norm.weight.data.cpu().numpy()
        new_weights[:] = old_weights[[i for i in range(old_weights.shape[0]) if i not in filter_index]]

        # 删除多余的bias
        old_bias=batch_norm.bias.data.cpu().numpy()
        new_bias=new_batch_norm.bias.data.cpu().numpy()
        new_bias[:] = old_bias[[i for i in range(old_bias.shape[0]) if i not in filter_index]]

        old_running_mean=batch_norm.running_mean.cpu().numpy()
        new_running_mean=new_batch_norm.running_mean.cpu().numpy()
        new_running_mean[:] = old_running_mean[[i for i in range(old_running_mean.shape[0]) if i not in filter_index]]

        old_running_var=batch_norm.running_var.cpu().numpy()
        new_running_var=new_batch_norm.running_var.cpu().numpy()
        new_running_var[:] = old_running_var[[i for i in range(old_running_var.shape[0]) if i not in filter_index]]

        if torch.cuda.is_available():
            new_batch_norm.cuda()
        model.features = torch.nn.Sequential(
            *(replace_layers(mod, [batch_norm], [new_batch_norm]) for mod in model.features))
    '''
    删除后继的Conv层的对应连接
    '''
    if next_conv is not None: # 如果后继的Conv不为空
        next_new_conv = torch.nn.Conv2d(in_channels=next_conv.in_channels - len(filter_index),
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=(next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()
        new_weights[:] = old_weights[:,[i for i in range(old_weights.shape[1]) if i not in filter_index],:,:]  # 复制剩余的filters的weight

        if next_conv.bias is not None:
            next_new_conv.bias.data = next_conv.bias.data
        if torch.cuda.is_available():
            next_new_conv.cuda()
        model.features=torch.nn.Sequential(                                               #生成替换为new_next_conv的features
            *(replace_layers(mod,[next_conv],[next_new_conv]) for mod in model.features))

    else:      # 如果后继的Conv层为空，即直接连接到全连接层
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        # layer_index = 0  # 所找到的层的序号
        print("正在剪枝Conv和FC的连接处")
        old_linear_layer = None  # 旧的线性层

        '''
        找到全连接层的第一个Linear层
        '''
        for _, module in model.classifier._modules.items():  # classifier是VGG中定义的成员

            if isinstance(module, torch.nn.Linear):  # 线性层
                old_linear_layer = module
                break
            # layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")

        '''
        原VGG网络中，old_linear_layer.in_features=512*7*7，因为最后一个Pool层的输出为7*7*512
        '''
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)  # 不知道为什么要引入这个变量

        '''确定Linear层的形状'''
        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - len(filter_index)*params_per_input_channel,
                            old_linear_layer.out_features)  # 线性层的输出不变

        '''保存weights'''
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        node_index=[]  # 要剪去的神经元
        for f in filter_index:
            node_index.extend([i for i in range(f*params_per_input_channel,(f+1)*params_per_input_channel)])

        new_weights[:] = old_weights[:,[i for i in range(old_weights.shape[1]) if i not in node_index]]  # 复制剩余的filters的weight

        #
        # new_weights[:, : filter_index * params_per_input_channel] = \
        #     old_weights[:, : filter_index * params_per_input_channel]
        # new_weights[:, filter_index * params_per_input_channel:] = \
        #     old_weights[:, (filter_index + 1) * params_per_input_channel:]

        '''拷贝原来的bias，因为输出的shape不变'''
        new_linear_layer.bias.data = old_linear_layer.bias.data

        if torch.cuda.is_available():
            new_linear_layer.cuda()

        '''用新的FC层代替旧的FC层'''
        model.classifier = torch.nn.Sequential(
            *(replace_layers(mod, [old_linear_layer], [new_linear_layer]) for mod in model.classifier))
    return model


def select_and_prune_filter(model,ord,layer_index=0,num_to_prune=0,percent_of_pruning=0):
    """
    选择卷积层，并剪枝
    :param model: VGG或ResNet模型，net model
    :param ord: 整数，which norm to compute as the standard. Support l1 and l2 norm
    :param layer_index: 整数，layer in which the filters being pruned. If being set to 0, all conv layers will be pruned.
    :param num_to_prune: 整数，要被剪枝的卷积核数目。number of filters to prune. Disabled if percent_of_pruning is not 0
    :param percent percent_of_pruning: 压缩率。percent of filters to prune for one conv
    :return: filter indexes in the [layer_index] layer
    """

    if ord!=1 and ord !=2:  # 判断是否输入错误
        raise TypeError('unsupported type of norm')


    '''
    遍历所有层，查找要剪枝的层，保存在conv中
    '''
    i = 0  # 记录第i个卷积层
    conv_index=-1                           # index of the conv in model.features
    for mod in model.features:  # 针对VGG16_bn的Module有44个features（不计算全连接层）
        conv_index+=1           # 该Conv对应的是第几个feature，下标从0开始。
        if isinstance(mod, torch.nn.modules.conv.Conv2d): # 若是卷积层
            i += 1                          # 该Conv在层结构当中是第几层。
            if i == layer_index:            # 若是要剪枝的卷积层
                conv=mod                    # 获取要剪枝的卷积层
                break


    if percent_of_pruning is not 0:
        # 如果要被剪枝的卷积核数目不为0，则失效
        if num_to_prune is not 0:
            print('Warning: Param: num_to_prune disabled!')
        num_to_prune=int(conv.out_channels * percent_of_pruning)  # 计算要剪枝的通道数

    '''
    获取要剪枝的层的weight
    '''
    weights = model.features[conv_index].weight.data.cpu().numpy()

    '''
    计算该层每一个filter的范数，并将范数从小到大排列，选取最小的num_prune个filter，对其进行修剪。
    '''
    filter_norm=np.linalg.norm(weights,ord=ord,axis=(2,3))
    if ord==1:  # 采用L1范数
        filter_norm=np.sum(filter_norm,axis=1)
    elif ord==2:  # 采用L2范数
        filter_norm=np.square(filter_norm)
        filter_norm=np.sum(filter_norm,axis=1)

    # 将权重的范数从小到大排列
    filter_min_norm_index=np.argsort(filter_norm)  # np.argsort(x)是将x中的元素从小到大排列，提取其对应的index(索引)

    # 对卷积层剪枝
    model=prune_conv_layer(model,
                           layer_index,
                           filter_min_norm_index[:num_to_prune])  # 选取最小的num_prune个filter

    return model


def prune_fc_layer(model, layer_index, sparsity):
    """
    对全连接层剪枝（矩阵稀疏化）
    :param model: 定义的模型，ResNet或VGG
    :param layer_index: 整数，减去的是第几个全连接层(对于VGG而言，layer_index只能为1或2,因最后一层不可剪枝)
    :param sparsity: 整数，稀疏率
    :return:
    """
    '''确定要剪枝的Linear层'''
    i=0
    old_fc=None  # 旧的FC层
    for mod in model.classifier:
        i+=1
        if i==layer_index:
            old_fc=mod
            break
        else:
            continue

    new_fc = torch.nn.Linear(old_fc.in_features, old_fc.out_features)  # 确定新的Linear层的形状

    old_weights = old_fc.weight.data.cpu().numpy()
    new_weights=new_fc.weight.data.cpu().numpy()

    '''矩阵稀疏化'''
    new_weights_abs=torch.abs(new_fc.weight)  # 对一个Tensor求每一个元素的绝对值
    threshold=np.percentile(new_weights_abs.data.cpu().numpy(), sparsity)  # 确定阈值
    mask=np.abs(old_weights)>threshold  # 得到一个Bool型的矩阵
    temp=old_weights*mask
    new_weights[:]=temp[:]  # 获得一个剪枝后的矩阵

    '''用新的FC层代替旧的FC层'''
    model.classifier = torch.nn.Sequential(
        *(replace_layers(mod, [old_fc], [new_fc]) for mod in model.classifier))

    return model


if __name__ == "__main__":
    model= vgg.vgg16_bn(pretrained=True)
    # select_and_prune_filter(model,layer_index=3,num_to_prune=2,ord=2)
    # prune_conv_layer(model,layer_index=3,filter_index=[0,1,2])
    prune_fc_layer(model,layer_index=1,sparsity=50)
    # print(model.classifier[0].weight.data.cpu().numpy())