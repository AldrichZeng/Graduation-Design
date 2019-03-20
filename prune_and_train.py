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
from prune import select_and_prune_filter

def evaluate_model(net,
                   data_loader,
                   save_model,
                   checkpoint_path=None,
                   highest_accuracy_path=None,
                   global_step_path=None,
                   global_step=0,
                   ):
    '''
    评估模型
    :param net: 定义的模型
    :param data_loader: 测试集的加载器
    :param save_model: Boolean. Whether or not to save the model.
    :param checkpoint_path: 
    :param highest_accuracy_path: 
    :param global_step_path: 
    :param global_step: global step of the current trained model
    '''
    if save_model:
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise AttributeError('checkpoint path is wrong')
        if highest_accuracy_path is None or not os.path.exists(highest_accuracy_path):
            raise AttributeError('highest_accuracy path is wrong')
        if global_step_path is None or not os.path.exists(global_step_path):
            raise AttributeError('global_step path is wrong')
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("{} Start Evaluation".format(datetime.now()))
    print("{} global step = {}".format(datetime.now(), global_step))

    # 报错：在global_step=0时停止
    with torch.no_grad():
        correct = 0
        total = 0
        for val_data in data_loader:
            net.eval()
            images, labels = val_data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        correct = float(correct.cpu().numpy().tolist())
        accuracy = correct / total
        print("{} Accuracy = {:.4f}".format(datetime.now(), accuracy))
        if save_model and accuracy > highest_accuracy:
            highest_accuracy = accuracy
            # save model
            print("{} Saving model...".format(datetime.now()))
            torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
            print("{} Model saved ".format(datetime.now()))
            # save highest accuracy
            f = open(highest_accuracy_path, 'w')
            f.write(str(highest_accuracy))
            f.close()
            # save global step
            f = open(global_step_path, 'w')
            f.write(str(global_step))
            print("{} model saved at global step = {}".format(datetime.now(), global_step))
            f.close()

def prune_and_train(
                    model_name,
                    pretrained=False,
                    dataset_name='imagenet',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=None,
                    highest_accuracy_path=None,
                    global_step_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    percent_of_pruning=0.3,
                    ord=2,
                  ):
    """
    剪枝并训练。根据model_name定义模型，调用该模型的进行剪枝，并进行训练。
    implemented according to "Pruning Filters For Efficient ConvNets" by Hao Li
    :param model_name: 字符串，模型的名字
    :param pretrained: Bool
    :param dataset_name: 数据集名称，默认为"ImageNet"
    :param learning_rate: Float，学习率
    :param num_epochs:  整数，迭代次数
    :param batch_size:
    :param checkpoint_step:
    :param checkpoint_path:
    :param highest_accuracy_path:
    :param global_step_path:
    :param default_image_size:
    :param momentum:
    :param num_workers:
    :param percent_of_pruning:
    :param ord:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    temp=re.search(r'(\d+)',model_name).span()[0]   # 正则表达式：\d匹配数字，+表示任意位数，用于匹配一个数字
    #  span() 返回一个元组包含匹配 (开始,结束) 的位置
    model=model_name[:temp]     # 模型的名称，如vgg,resnet。model是一个字符串

    del temp
    print(globals()[model])
    # 定义模型
    net=getattr(globals()[model],model_name)(pretrained=pretrained).to(device)  # 相当于调用vgg16_bn()函数
    # getattr() 函数用于返回一个对象属性值。
    # globals() 函数会以字典类型返回当前位置的全部全局变量。

    # 计算卷积层的数目
    num_conv=0                      # num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv+=1

    # 遍历所有的卷积层，进行剪枝
    # select_and_prune_filter在prune.py中定义
    for i in range(1,num_conv+1):
        net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=percent_of_pruning, ord=ord)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    # 准备数据
    if dataset_name is 'imagenet':
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_path=conf.imagenet['validation_set_path']

    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 载入数据
    # train = datasets.ImageFolder(train_set_path, transform)
    train=datasets.CIFAR10("./data",download=True,transform=transform)
    # val = datasets.ImageFolder(validation_set_path, transform)
    val=datasets.CIFAR10("./data",download=True,transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if checkpoint_path is None:
        checkpoint_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/checkpoint'
    if highest_accuracy_path is None:
        highest_accuracy_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/highest_accuracy.txt'
    if global_step_path is None:
        global_step_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/global_step.txt'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    if  os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')  # 读取文件
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)
        del highest_accuracy

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')  # 读取文件
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))
    else:
        print('{} test the model after pruned'.format(datetime.now()))    # 无之前保存的模型
        evaluate_model(net,validation_loader,save_model=False)  # 评估模型

        # 报错：在此处停止

    # 开始训练
    print("{} Start training ".format(datetime.now())+model_name+"...")
    for epoch in range(math.floor(global_step*batch_size/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if global_step / math.ceil(train_set_size / batch_size)==epoch+1:   #one epoch of training finished
                evaluate_model(net,validation_loader,
                               save_model=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               global_step_path=global_step_path,
                               global_step=global_step)
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # 前向传播
            outputs = net(images)
            # 计算loss
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 单次优化
            optimizer.step()
            global_step += 1

            if step % checkpoint_step == 0 and step != 0:
                evaluate_model(net,validation_loader,
                               save_model=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               global_step_path=global_step_path,
                               global_step=global_step)
                print('{} continue training'.format(datetime.now()))


if __name__ == "__main__":
    prune_and_train(model_name='vgg16_bn',
                    pretrained=True,
                    checkpoint_step=5000,
                    percent_of_pruning=0.1,
                    num_epochs=20)