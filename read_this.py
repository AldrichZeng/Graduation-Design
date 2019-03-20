import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import vgg
import os
import matplotlib.pyplot as plt
from datetime import datetime
def train_resnet_18(
                    dataset_name='imagenet',
                    prune=False,
                    prune_params='',
                    learning_rate=0.01,
                    num_epochs=20,
                    batch_size=32,
                    # checkpoint_step=500,
                    # checkpoint_path=conf.root_path+'resnet_18'+conf.checkpoint_path,
                    # highest_accuracy_path=conf.root_path+'resnet_18'+conf.highest_accuracy_path,
                    # global_step_path=conf.root_path+'resnet_18'+conf.global_step_path,
                    default_image_size=224,
                    num_workers=0
                  ):
    #定义数据集的一些信息
    if dataset_name is 'imagenet':
        train_set_size=conf.imagenet['train_set_size']
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        validation_set_path=conf.imagenet['validation_set_path']

    # Data loading code
    #定义怎么读取数据集

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

    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    # 训练集
    # train = datasets.ImageFolder(train_set_path, transform)
    train = datasets.CIFAR10("./data",download=True,transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 验证集
    # val = datasets.ImageFolder(validation_set_path, transform)
    val =datasets.CIFAR10("./data",download=True,transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #定义模型，调用了已经定义好的模型
    net = resnet.resnet18(pretrained=False)

    print("stop")

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


    print("{} Start training Resnet-18...".format(datetime.now()))
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        net.train()

        # 对train_loader中的数据进行迭代
        for step, data in enumerate(train_loader, 0):
            # 准备数据
            images, labels = data

            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)  # forward
            loss = criterion(outputs, labels)  # Loss function
            loss.backward()  # backward
            optimizer.step()  # 单步优化

            #训练100个 batch 就验证一下当前模型的准确率
            if step % 100 == 0 and step!=0:
                print("{} Start validation".format(datetime.now()))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for val_data in validation_loader:
                        net.eval()
                        images, labels = val_data
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        print(outputs.shape)
                        _, predicted = torch.max(outputs.data, 1)
                        print(predicted)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct = float(correct.cpu().numpy().tolist())
                    accuracy =  correct / total
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))

train_resnet_18()