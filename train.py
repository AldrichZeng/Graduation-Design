import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import vgg
import os
from datetime import datetime
from prune import select_and_prune_filter


#todo:可以改写为根据字符串创建对应模型，参见prune_and_train中，等有时间记得改过来

def exponential_decay_learning_rate(optimizer, learning_rate, global_step, decay_steps,decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate *decay_rate ** int(global_step / decay_steps)
    if lr!=learning_rate:
        print('--------------------------------------------------------------------------------')
        print('learning rate at present is %f'%lr)
        print('--------------------------------------------------------------------------------')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_resnet_18(
                    dataset_name='imagenet',
                    prune=False,
                    prune_params='',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay,
                    num_epochs_per_decay=conf.num_epochs_per_decay,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=conf.root_path+'resnet_18'+conf.checkpoint_path,
                    highest_accuracy_path=conf.root_path+'resnet_18'+conf.highest_accuracy_path,
                    global_step_path=conf.root_path+'resnet_18'+conf.global_step_path,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers
                  ):
    if dataset_name is 'imagenet':
        train_set_size=conf.imagenet['train_set_size']
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        validation_set_path=conf.imagenet['validation_set_path']

    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    train = datasets.ImageFolder(train_set_path, transform)
    val = datasets.ImageFolder(validation_set_path, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #define the model
    net = resnet.resnet18(True).to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    highest_accuracy = 0
    if os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))

    print("{} Start training Resnet-18...".format(datetime.now()))
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()

        print("{} Start validation".format(datetime.now()))
        print("{} global step = {}".format(datetime.now(), global_step))
        with torch.no_grad():
            correct = 0
            total = 0
            for val_data in validation_loader:
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
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))

        #one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #decay learning rate
            global_step+=1
            decay_steps = int(train_set_size / batch_size * num_epochs_per_decay)
            exponential_decay_learning_rate(optimizer,learning_rate,global_step,decay_steps,learning_rate_decay_factor)

            if step % checkpoint_step == 0 and step!=0:
                print("{} Start validation".format(datetime.now()))
                print("{} global step = {}".format(datetime.now(), global_step))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for val_data in validation_loader:
                        net.eval()
                        images, labels = val_data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct = float(correct.cpu().numpy().tolist())
                    accuracy =  correct / total
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))
                    if accuracy>highest_accuracy:
                        highest_accuracy=accuracy
                        #save model
                        print("{} Saving model...".format(datetime.now()))
                        torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
                        print("{} Model saved ".format(datetime.now()))
                        #save highest accuracy
                        f = open(highest_accuracy_path, 'w')
                        f.write(str(highest_accuracy))
                        f.close()
                        #save global step
                        f=open(global_step_path,'w')
                        f.write(str(global_step))
                        print("{} model saved at global step = {}".format(datetime.now(), global_step))
                        f.close()
                        print('continue training')

def train_resnet_50(
                    pretrained=False,
                    dataset_name='imagenet',
                    prune=False,
                    prune_params='',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay,
                    num_epochs_per_decay=conf.num_epochs_per_decay,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=conf.root_path+'resnet_50'+conf.checkpoint_path,
                    highest_accuracy_path=conf.root_path+'resnet_50'+conf.highest_accuracy_path,
                    global_step_path=conf.root_path+'resnet_50'+conf.global_step_path,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers
                  ):
    if dataset_name is 'imagenet':
        train_set_size=conf.imagenet['train_set_size']
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        validation_set_path=conf.imagenet['validation_set_path']

    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    train = datasets.ImageFolder(train_set_path, transform)
    val = datasets.ImageFolder(validation_set_path, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #define the model
    net = resnet.resnet50(pretrained).to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    highest_accuracy = 0
    if os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        if os.path.exists(model_saved_at):
            print('load model from'+model_saved_at)
            net.load_state_dict(torch.load(model_saved_at))

    print("{} Start training Resnet-50...".format(datetime.now()))
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()

        #one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            # 准备数据
            length = len(train_loader)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #decay learning rate
            global_step+=1
            decay_steps = int(train_set_size / batch_size * num_epochs_per_decay)
            exponential_decay_learning_rate(optimizer,learning_rate,global_step,decay_steps,learning_rate_decay_factor)

            if step % checkpoint_step == 0 and step!=0:
                print("{} Start validation".format(datetime.now()))
                print("{} global step = {}".format(datetime.now(), global_step))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for val_data in validation_loader:
                        net.eval()
                        images, labels = val_data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct = float(correct.cpu().numpy().tolist())
                    accuracy =  correct / total
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))
                    if accuracy>highest_accuracy:
                        highest_accuracy=accuracy
                        #save model
                        print("{} Saving model...".format(datetime.now()))
                        torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
                        print("{} Model saved ".format(datetime.now()))
                        #save highest accuracy
                        f = open(highest_accuracy_path, 'w')
                        f.write(str(highest_accuracy))
                        f.close()
                        #save global step
                        f=open(global_step_path,'w')
                        f.write(str(global_step))
                        print("{} model saved at global step = {}".format(datetime.now(), global_step))
                        f.close()
                        print("{} Continue Training".format(datetime.now()))


def train_resnet_101(
                    dataset_name='imagenet',
                    prune=False,
                    prune_params='',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay,
                    num_epochs_per_decay=conf.num_epochs_per_decay,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=conf.root_path+'resnet_101'+conf.checkpoint_path,
                    highest_accuracy_path=conf.root_path+'resnet_101'+conf.highest_accuracy_path,
                    global_step_path=conf.root_path+'resnet_101'+conf.global_step_path,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers
                  ):
    if dataset_name is 'imagenet':
        train_set_size=conf.imagenet['train_set_size']
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        validation_set_path=conf.imagenet['validation_set_path']

    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    train = datasets.ImageFolder(train_set_path, transform)
    val = datasets.ImageFolder(validation_set_path, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #define the model
    net = resnet.resnet101(True).to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    highest_accuracy = 0
    if os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))

    print("{} Start training Resnet-101...".format(datetime.now()))
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()

        #one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            # 准备数据
            length = len(train_loader)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #decay learning rate
            global_step+=1
            decay_steps = int(train_set_size / batch_size * num_epochs_per_decay)
            exponential_decay_learning_rate(optimizer,learning_rate,global_step,decay_steps,learning_rate_decay_factor)

            if step % checkpoint_step == 0 and step!=0:
                print("{} Start validation".format(datetime.now()))
                print("{} global step = {}".format(datetime.now(), global_step))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for val_data in validation_loader:
                        net.eval()
                        images, labels = val_data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct = float(correct.cpu().numpy().tolist())
                    accuracy =  correct / total
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))
                    if accuracy>highest_accuracy:
                        highest_accuracy=accuracy
                        #save model
                        print("{} Saving model...".format(datetime.now()))
                        torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
                        print("{} Model saved ".format(datetime.now()))
                        #save highest accuracy
                        f = open(highest_accuracy_path, 'w')
                        f.write(str(highest_accuracy))
                        f.close()
                        #save global step
                        f=open(global_step_path,'w')
                        f.write(str(global_step))
                        print("{} model saved at global step = {}".format(datetime.now(), global_step))
                        f.close()
                        print('continue training')

def train_vgg_11(
                    pretrained=False,
                    dataset_name='imagenet',
                    prune=False,
                    prune_params='',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay,
                    num_epochs_per_decay=conf.num_epochs_per_decay,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=conf.root_path+'vgg_11'+conf.checkpoint_path,
                    highest_accuracy_path=conf.root_path+'vgg_11'+conf.highest_accuracy_path,
                    global_step_path=conf.root_path+'vgg_11'+conf.global_step_path,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers
                  ):
    if dataset_name is 'imagenet':
        train_set_size=conf.imagenet['train_set_size']
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        validation_set_path=conf.imagenet['validation_set_path']

    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))



    #define the model
    net = vgg.vgg11(pretrained).to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    highest_accuracy = 0
    if os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))



        # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    train = datasets.ImageFolder(train_set_path, transform)
    val = datasets.ImageFolder(validation_set_path, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("{} Start training vgg-11...".format(datetime.now()))
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()

        #one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            # 准备数据
            length = len(train_loader)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #decay learning rate
            global_step+=1
            decay_steps = int(train_set_size / batch_size * num_epochs_per_decay)
            exponential_decay_learning_rate(optimizer,learning_rate,global_step,decay_steps,learning_rate_decay_factor)

            if step % checkpoint_step == 0 and step!=0:
                print("{} Start validation".format(datetime.now()))
                print("{} global step = {}".format(datetime.now(), global_step))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for val_data in validation_loader:
                        net.eval()
                        images, labels = val_data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct = float(correct.cpu().numpy().tolist())
                    accuracy =  correct / total
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))
                    if accuracy>highest_accuracy:
                        highest_accuracy=accuracy
                        #save model
                        print("{} Saving model...".format(datetime.now()))
                        torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
                        print("{} Model saved ".format(datetime.now()))
                        #save highest accuracy
                        f = open(highest_accuracy_path, 'w')
                        f.write(str(highest_accuracy))
                        f.close()
                        #save global step
                        f=open(global_step_path,'w')
                        f.write(str(global_step))
                        print("{} model saved at global step = {}".format(datetime.now(), global_step))
                        f.close()
                        print('continue training')

if __name__ == "__main__":
    train_vgg_11(pretrained=True)

