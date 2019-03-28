import torch
import torch.nn



def print_net_weights(net,conv_layer_index=[],fc_layer_index=[]):
    for index in conv_layer_index:
        if index > 2 or index<1:
            print("错误，无法打印第"+index+"个Conv层")
            continue
        print("第"+str(index)+"个卷积层")
        print_layer_weight(net,conv_layer_index=index,conv=True)

    for index in fc_layer_index:
        if index>2 or index<1:
            print("错误，无法打印第" + index + "个卷FC层")
            continue
        print("第" + str(index) + "个全连接层")
        print_layer_weight(net,fc_layer_index=index,fc=True)


def print_layer_weight(net,conv_layer_index=0,conv=True,fc_layer_index=0,fc=False):
    if conv:
        i=0
        for mod in net.features:
            if isinstance(mod,torch.nn.modules.conv.Conv2d):
                i+=1
                if i==conv_layer_index:
                    # 找到了要打印的卷积层
                    print(mod.weight.data.cpu().numpy())
    if fc:
        i=0
        for mod in net.classifier:
            if isinstance(mod,torch.nn.Linear):
                i+=1
                if i==fc_layer_index:
                    # 找到了要打印的全连接层
                    print(mod.weight.data.cpu().numpy())

def print_net_shape(net,conv_layer_index,fc_layer_index):
    for index in conv_layer_index:
        if index > 13 or index<=0:
            print("错误，无法打印第"+index+"个Conv层")
            continue
        print_layer_shape(net,conv_layer_index=index,conv=True)

    for index in fc_layer_index:
        if index>2 or index<=0:
            print("错误，无法打印第" + index + "个卷FC层")
            continue
        print_layer_shape(net,fc_layer_index=index,fc=True)


def print_layer_shape(net,conv_layer_index=0,conv=True,fc_layer_index=0,fc=False):
    if conv:
        i=0
        for mod in net.features:
            if isinstance(mod,torch.nn.modules.conv.Conv2d):
                i+=1
                if i==conv_layer_index:
                    # 找到了要打印的卷积层
                    print(mod.weight.data.cpu().numpy().shape)
    if fc:
        i=0
        for mod in net.classifier:
            if isinstance(mod,torch.nn.Linear):
                i+=1
                if i==fc_layer_index:
                    # 找到了要打印的全连接层
                    print(mod.weight.data.cpu().numpy().shape)


