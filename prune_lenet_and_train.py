import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import resnet
import vgg
import os
import re
from datetime import datetime
from prune import select_and_prune_fc,select_and_prune_conv,replace_layers
import helper
import LeNet
from train_LeNet5_on_MNIST import get_globalStep_From_File,get_highestAccuracy_From_File,save_Model_and_StateDict
BATCH_SIZE=64
NUM_EPOCH=100
ROOT = "save_and_load/lenet5/"  # 根目录（文件中未判断是否存在，请直接建立）

def evaluate_model(net,
                   data_loader,
                   global_step=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("{} 开始测试, global step = {}".format(datetime.now(), global_step))

    with torch.no_grad():
        correct = 0
        total = 0
        for i,val_data in enumerate(data_loader):
            net.eval()
            images, labels = val_data  # images是一个32*3*32*32的tensor，第一个32是batch_size？
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        correct = float(correct.cpu().numpy().tolist())
        accuracy = correct / total
        print("{} 测试集准确率 Accuracy = {:.6f}".format(datetime.now(), accuracy))
        highest_accuracy = accuracy
        # save_Model_and_StateDict(net, global_step=global_step, highest_accuracy=highest_accuracy)  # 保存模型
        # save_prune_result(net, global_step=global_step, sparsity=sparsity, accuracy=highest_accuracy)
    return highest_accuracy,global_step

def prune_lenet_and_train(learning_rate=conf.learning_rate,
                          fc_layer_list=[1,2],
                          conv_layer_list=[1,2],
                          sparsity=90):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    '''定义模型'''
    net=LeNet.lenet5().to(device)  # 相当于调用vgg16_bn()函数
    highest_accuracy=get_highestAccuracy_From_File()
    global_step,net=get_globalStep_From_File(net,fileName="global_step.txt")

    # '''剪枝Conv层，生成新的模型'''
    # num_conv=0                      # 用于计算卷积层的数目（LeNet中有2个卷积层）
    # for mod in net.features:
    #     if isinstance(mod, torch.nn.modules.conv.Conv2d):
    #         num_conv+=1
    # for i in range(1,num_conv+1):
    #     net = select_and_prune_filter(net,
    #                                   layer_index=i, # 对第i个卷积层进行剪枝
    #                                   percent_of_pruning=percent_of_pruning, # 剪枝率
    #                                   ord=ord)# 范数

    '''稀疏化Conv层，生成新的模型'''
    net,conv_mask_dict=select_and_prune_conv(net,layer_list=conv_layer_list,sparsity=sparsity)
    '''稀疏化FC层，生成新的模型'''
    # net,fc_mask_dict=select_and_prune_fc(net,layer_list=fc_layer_list,sparsity=sparsity)  # LeNet5有3个FC层，只能对前两个剪枝
    '''定义损失函数'''
    criterion = nn.CrossEntropyLoss()  # 交叉熵
    '''定义优化器'''
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    '''准备数据(将所有输入都统一为224*224)'''
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)  # 定义训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 定义训练批处理数据
    testset = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)  # 定义测试数据集
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)  # 定义测试批处理数据

    print('{} 剪枝完毕，测试准确率 '.format(datetime.now()))    # 无之前保存的模型
    highest_accuracy,global_step=evaluate_model(net,validation_loader,highest_accuracy=highest_accuracy,global_step=global_step)  # 评估模型
    # 注意，highest_accuracy为剪枝后当前的最高准确率，不是全局（不包含剪枝前）
    save_prune_result(net, global_step=global_step, sparsity=sparsity, accuracy=highest_accuracy)

    print("{} Start Retraining LeNet5...".format(datetime.now()))
    for epoch in range(0,NUM_EPOCH):
        sum_loss=0
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()  # 设为训练模式
        for step, data in enumerate(train_loader, 0):
            net.train()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            '''这里是retrain，因此需要对Conv层和FC层做一些改动'''
            net=replace_conv_weights(net,conv_mask_dict=conv_mask_dict)
            # net=replace_fc_weights(net, fc_mask_dict=fc_mask_dict)
            # helper.print_net_weights(net,conv_layer_index=[1,2],fc_layer_index=[])
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            sum_loss += loss.item()
            if step % 100 == 99:  # 每训练100个batch就打印loss
                print('[%d, %d] loss: %.03f' % (epoch + 1, step + 1, sum_loss / 100))
                sum_loss = 0.0
        print("{} Start validation, global step = {}".format(datetime.now(), global_step))
        with torch.no_grad():  # 每运行完一个epoch就进行验证（Validation）
            correct = 0
            total = 0
            for data in validation_loader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类
                total += labels.size(0)
                correct += (predicted == labels).sum()
            correct = float(correct.cpu().numpy().tolist())
            accuracy = correct / total
            print("{} 验证集准确率 Accuracy = {:.6f}".format(datetime.now(), accuracy))
            if accuracy > highest_accuracy:  # 如果有更高的准确率，则保存模型
                highest_accuracy = accuracy
                save_prune_result(net,global_step=global_step,sparsity=sparsity,accuracy=highest_accuracy)


def save_prune_result(net,global_step,sparsity,accuracy,root=ROOT):
    PathForStateDict = '%s/stateDict_%d_%d.pth' % (root + "checkpoints_prune_conv", global_step, sparsity)
    # PathForStateDict = '%s/stateDict_%d_%d.pth' % (ROOT + "checkpoints_prune_fc", global_step, sparsity)
    torch.save(net.state_dict(),PathForStateDict)

    f=open(root+"accuracy_prune.txt","w")
    f.write(str(accuracy))
    f.close()

    f=open(root+"globalStep_prune.txt","w")
    f.write(str(global_step))
    f.close()

def replace_fc_weights(model, fc_mask_dict):
    """
    替换FC层的权重,用于对retrain过程中的W矩阵替换为剪枝后的矩阵
    :return:
    """
    for index in fc_mask_dict:  # 遍历所有的{layer_index:mask}
        # 查找需要替换的FC层
        old_fc=None
        i=0
        for mode in model.classifier:
            if isinstance(mode,torch.nn.Linear):
                i+=1
                if i==index:
                    old_fc=mode
                    break
        if old_fc is not None:
            new_fc=torch.nn.Linear(old_fc.in_features, old_fc.out_features)

            # 改变权重值
            old_weights=old_fc.weight.data.cpu().numpy()
            new_weights=new_fc.weight.data.cpu().numpy()

            temp=old_weights[:]
            new_weights[:]=temp[:] * fc_mask_dict[index]  # 恢复
        if torch.cuda.is_available():
            new_fc.cuda()
        model.classifier = torch.nn.Sequential(
            *(replace_layers(mode, [old_fc], [new_fc]) for mode in model.classifier))
    return model

def replace_conv_weights(model,conv_mask_dict):
    for index in conv_mask_dict:  # 遍历所有的{layer_index:mask}
        # 查找需要替换的Conv层
        old_conv=None
        i=0
        for mod in model.features:
            if isinstance(mod,torch.nn.modules.conv.Conv2d):
                i+=1
                if i==index:
                    old_conv=mod
                    break
        if old_conv is not None:
            new_conv=torch.nn.Conv2d(in_channels=old_conv.in_channels,
                                   out_channels=old_conv.out_channels,
                                   kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride,
                                   padding=old_conv.padding,
                                   dilation=old_conv.dilation,
                                   groups=old_conv.groups,
                                   bias=(old_conv.bias is not None))

            # 改变权重值
            old_weights=old_conv.weight.data.cpu().numpy()
            new_weights=new_conv.weight.data.cpu().numpy()

            temp=old_weights[:]
            print(temp)
            new_weights[:]=temp[:] * conv_mask_dict[index]  # 恢复
            print(new_weights)
        if torch.cuda.is_available():
            new_conv.cuda()
        model.features = torch.nn.Sequential(
            *(replace_layers(mode, [old_conv], [new_conv]) for mode in model.features))
    return model

if __name__ == "__main__":
    prune_lenet_and_train()