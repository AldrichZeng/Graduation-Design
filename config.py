import numpy as np

#training params
num_epochs=10                       #times for the use of all training data
batch_size=32                       #number of images for one batch
learning_rate=0.001
learning_rate_decay_factor=0.94     #decay factor for exponential decay
weight_decay=5e-4                   # weight decay (L2 penalty)
num_epochs_per_decay=2.5
dropout_rate=0.5
momentum=0.9

#dataset processing params
num_workers=8


#dataset params
#imagenet
imagenet=dict()
imagenet['num_class']=1001                                          #number of the classes
imagenet['label_offset']=1                                          #offset of the label
imagenet['mean']=[0.485, 0.456, 0.406]
imagenet['std']=[0.229, 0.224, 0.225]
imagenet['train_set_size']=1271167
imagenet['validation_set_size']=50000
# ImageNet数据集在本地没有，可以使用CIFAR训练
imagenet['train_set_path']='/home/victorfang/Desktop/imagenet所有数据/imagenet_train'
imagenet['validation_set_path']='/home/victorfang/Desktop/imagenet所有数据/imagenet_validation'


#model saving params
#how often to write summary and checkpoint
checkpoint_step=4000

# Path for tf.summary.FileWriter and to store model checkpoints
root_path='/home/victorfang/Desktop/pytorch_model/'
checkpoint_path = "_model_saved/checkpoints"
highest_accuracy_path='_model_saved/accuracy.txt'
global_step_path='_model_saved/global_step.txt'
epoch_path='_model_saved/epoch.txt'