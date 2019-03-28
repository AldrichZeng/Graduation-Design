import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import LeNet
import os
from datetime import datetime

# 超参数设置
EPOCH = 100  # 遍历数据集次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率
ROOT="save_and_load/lenet5/"  # 根目录（文件中未判断是否存在，请直接建立）

def get_highestAccuracy_From_File(root=ROOT, fileName="accuracy.txt"):
    path=root+fileName
    highest_accuracy = 0
    if os.path.exists(path):  # 若存在该文件
        f = open(path, 'r')  # 只读方式打开文件（txt文件）
        highest_accuracy = float(f.read())  # 强制转换为float
        f.close()  # 关闭文件
        print('highest accuracy from previous training is %f' % highest_accuracy)
    print("读入最高精确度 highest_accuracy = " + str(highest_accuracy))
    return highest_accuracy

def get_globalStep_From_File(net, root=ROOT, fileName="global_step.txt"):
    path = root + fileName
    global_step = 0
    if os.path.exists(path):  # 若存在文件
        f = open(path, 'r')  # 只读方式打开文件（txt文件）
        global_step = int(f.read())  # 强制转换问int
        f.close()  # 关闭文件
        print("读入全局迭代数 global_step = " + str(global_step))
        #todo: 如果要读取剪枝后的模型，请修改路径
        state_dict_saved_at = root + 'checkpoints/stateDict_' + str(global_step) + '.pth'  # 模型参数保存的路径
        # model_save_at=root + 'checkpoints/model_' + str(global_step) + '.pth'  # 模型结构保存的路径
        # net = torch.load(model_save_at)
        if os.path.exists(state_dict_saved_at):
            print('load model from file \"' + state_dict_saved_at+"\" ")
            net.load_state_dict(torch.load(state_dict_saved_at))  # 加载模型的state_dict
    return global_step,net


def save_Model_and_StateDict(net, global_step, highest_accuracy):
    """
    保存模型的结构及其参数信息
    """
    print("{} 正在保存模型及其参数".format(datetime.now()))

    PathForStateDict = '%s/stateDict_%d.pth' % (ROOT + "checkpoints", global_step)
    torch.save(net.state_dict(), PathForStateDict)  # 保存模型的参数

    # PathForModel = '%s/model_%d.pth' % (ROOT + "checkpoints", global_step)
    # torch.save(net,PathForModel)  # 保存模型的结构

    f = open(ROOT + "accuracy.txt", 'w')  # 以写入方式打开文件
    f.write(str(highest_accuracy))  # 写入文件：保存最高精确度
    f.close()  # 关闭文件

    f = open(ROOT + "global_step.txt", 'w')  # 以写入方式打开文件
    f.write(str(global_step))  # 写入文件：保存全局迭代次数
    f.close()  # 关闭文件

    print("{}  保存完毕，此时 global_step={}".format(datetime.now(),global_step))



if __name__=="__main__":
    '''定义device'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    '''定义模型'''
    net = LeNet.lenet5().to(device)
    '''定义损失函数'''
    criterion = nn.CrossEntropyLoss()  # 交叉熵
    '''定义优化器'''
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    # todo:文件读入，如有不同配置，需要更改
    highest_accuracy=get_highestAccuracy_From_File()
    global_step,net=get_globalStep_From_File(net)

    '''加载数据集'''
    transform = transforms.ToTensor()
    trainset = tv.datasets.MNIST(root='./data/',train=True,download=True,transform=transform)  # 定义训练数据集
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)  # 定义训练批处理数据
    testset = tv.datasets.MNIST(root='./data/',train=False,download=True,transform=transform)  # 定义测试数据集
    testloader = torch.utils.data.DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)  # 定义测试批处理数据
    # save_Model_and_StateDict(global_step=global_step, highest_accuracy=highest_accuracy)
    ''''''
    for epoch in range(1,EPOCH):
        print("正在进行第 "+str(epoch)+" 次迭代...")
        sum_loss = 0.0
        '''开始训练'''
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算Loss
            loss.backward()   # 反向传播
            optimizer.step()  # 单次优化
            sum_loss += loss.item()
            # 每训练100个batch打印一次平均loss
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

        # 每运行完一个epoch测试一下准确率
        print("{} Start validation, global step ={}".format(datetime.now(),global_step))
        with torch.no_grad():
            correct = 0
            total = 0
            '''开始测试'''
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类
                '''计算准确度'''
                total += labels.size(0)
                correct += (predicted == labels).sum()
            correct = float(correct.cpu().numpy().tolist())
            accuracy = correct / total
            print("{} 验证集 Accuracy = {:.4f}".format(datetime.now(), accuracy))
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                global_step = epoch+global_step
                save_Model_and_StateDict(net, global_step=global_step, highest_accuracy=highest_accuracy)



