import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import LeNet
from train_LeNet5_on_MNIST import get_globalStep_From_File,get_highestAccuracy_From_File,save_Model_and_StateDict
from torch.autograd import Variable
from decompositions import cp_decomposition_conv_layer
import tensorly as tl

BATCH_SIZE=64
NUM_EPOCH=100
ROOT = "save_and_load/lenet5/"  # 根目录（文件中未判断是否存在，请直接建立）
BATCH_SIZE = 64
transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)  # 定义训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 定义训练批处理数据
testset = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)  # 定义测试数据集
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)  # 定义测试批处理数据


def CPdecompose_and_retrain_lenet(learning_rate=0.001,
                                  checkpoints_path="checkpoints_prune_conv"):
    tl.set_backend('pytorch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    '''加载模型'''
    net=LeNet.lenet5().to(device)  # 相当于调用vgg16_bn()函数
    # highest_accuracy=get_highestAccuracy_From_File(fileName="accuracy_prune.txt")
    # global_step,net=get_globalStep_From_File(net,fileName="globalStep_prune.txt",check_point_path=checkpoints_path,h_accuracy="_98")
    global_step,net=get_globalStep_From_File(net,check_point_path=checkpoints_path,stateDict="stateDict_39113_90.pth")

    '''-----------------------------------CP分解-----------------------------------'''
    N = len(net.features._modules.keys())
    for i, key in enumerate(net.features._modules.keys()):
        if i >= N - 2:
            break
        if isinstance(net.features._modules[key], torch.nn.modules.conv.Conv2d):  # 如果是Conv层
            conv_layer = net.features._modules[key]  # 获取Conv层
            rank = max(conv_layer.weight.data.cpu().numpy().shape) // 3  # 计算rank
            decomposed = cp_decomposition_conv_layer(conv_layer, rank)  # CP分解

            net.features._modules[key] = decomposed

    for param in net.parameters():
        param.requires_grad = True
    net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # CP分解，使用较小的学习率

    '''-----------------------------------测试CP分解后的模型在test_set上的准确率-----------------------------------'''
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类
            total += labels.size(0)
            correct += (predicted == labels).sum()
        correct = float(correct.cpu().numpy().tolist())
        accuracy = float(correct) / total
    print("首次测试 Accuracy :", accuracy)

    '''-----------------------------------重训练，每次迭代完测试准确率-----------------------------------'''
    for epoch in range(100):
        print("Epoch: ", epoch+1)
        sum_loss = 0.0
        for i, data in enumerate(trainloader):
            net.train()
            net.zero_grad()
            optimizer.zero_grad()  # 梯度清零
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(net(inputs), Variable(labels))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if i % 100 == 99:  # 每训练100个batch就打印loss
                print('[%d, %d] loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类
                total += labels.size(0)
                correct += (predicted == labels).sum()
            correct = float(correct.cpu().numpy().tolist())
            accuracy = float(correct) / total
        print("Accuracy :", accuracy)


    torch.save(net, 'decomposed_finetuned_model.pth')


if __name__ == "__main__":

    CPdecompose_and_retrain_lenet()