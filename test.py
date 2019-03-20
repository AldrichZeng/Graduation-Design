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
import numpy as np
import config as conf


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
    if accuracy > highest_accuracy:
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