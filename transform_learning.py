import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

import model.alexnet as AlexNet
import model.resnet as ResNet
import model.vgg as Vgg
import model.lenet as LeNet

import os

device = torch.device('cuda')

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# CIFAR10 dataset 
train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True)

valid_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=256)

model = Vgg.VGG_2('VGG11')
print(model)

model_pre = Vgg.VGG('VGG11')
print(model_pre)

model = nn.DataParallel(model).cuda()
model_pre = nn.DataParallel(model_pre).cuda()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

# LeNet 
# if os.path.isfile('checkpoint/LeNet/0/model_best.pth.tar'):
#     print('=> loading checkpoint "{}"'.format('checkpoint/LeNet/0/model_best.pth.tar'))
#     checkpoint = torch.load('checkpoint/LeNet/0/model_best.pth.tar')
#     model_pre.load_state_dict(checkpoint['state_dict'])
#     model.module.conv1.load(model_pre.module.conv1)
#     model.module.conv2.load(model_pre.module.conv2)
#     model.module.fc1 = model_pre.module.fc1
#     model.module.fc2 = model_pre.module.fc2
#     model.module.fc3 = model_pre.module.fc3
# else:
#     print("=> no checkpoint found at '{}'".format('checkpoint/LeNet/0/model_best.pth.tar'))

# AlexNet 
# if os.path.isfile('checkpoint/AlexNet/1/model_best.pth.tar'):
#     print('=> loading checkpoint "{}"'.format('checkpoint/AlexNet/1/model_best.pth.tar'))
#     checkpoint = torch.load('checkpoint/AlexNet/1/model_best.pth.tar')
#     model_pre.load_state_dict(checkpoint['state_dict'])
#     model.module.features[0].load(model_pre.module.features[0])
#     model.module.features[4].load(model_pre.module.features[4])
#     model.module.features[8].load(model_pre.module.features[8])
#     model.module.features[11].load(model_pre.module.features[11])
#     model.module.features[14].load(model_pre.module.features[14])
#     model.module.classifier = model_pre.module.classifier
# else:
#     print("=> no checkpoint found at '{}'".format('checkpoint/AlexNet/1/model_best.pth.tar'))

if os.path.isfile('checkpoint/Vgg11/0/model_best.pth.tar'):
    print('=> loading checkpoint "{}"'.format('checkpoint/Vgg11/0/model_best.pth.tar'))
    checkpoint = torch.load('checkpoint//Vgg11/0/model_best.pth.tar')
    model_pre.load_state_dict(checkpoint['state_dict'])
    model.module.features[0].load(model_pre.module.features[0])
    model.module.features[3].load(model_pre.module.features[4])
    model.module.features[6].load(model_pre.module.features[8])
    model.module.features[8].load(model_pre.module.features[11])
    model.module.features[11].load(model_pre.module.features[15])
    model.module.features[13].load(model_pre.module.features[18])
    model.module.features[16].load(model_pre.module.features[22])
    model.module.features[18].load(model_pre.module.features[25])
    model.module.classifier = model_pre.module.classifier
else:
    print("=> no checkpoint found at '{}'".format('checkpoint/Vgg11/0/model_best.pth.tar'))

# # VGG13
# if os.path.isfile('checkpoint/Vgg13/0/model_best.pth.tar'):
#     print('=> loading checkpoint "{}"'.format('checkpoint/Vgg13/0/model_best.pth.tar'))
#     checkpoint = torch.load('checkpoint/Vgg13/0/model_best.pth.tar')
#     model_pre.load_state_dict(checkpoint['state_dict'])
#     model.module.features[0].load(model_pre.module.features[0])
#     model.module.features[1] = model_pre.module.features[1]
#     model.module.features[3].load(model_pre.module.features[3])
#     model.module.features[4] = model_pre.module.features[4]
#     model.module.features[7].load(model_pre.module.features[7])
#     model.module.features[8] = model_pre.module.features[8]
#     model.module.features[10].load(model_pre.module.features[10])
#     model.module.features[11] = model_pre.module.features[11]
#     model.module.features[14].load(model_pre.module.features[14])
#     model.module.features[15] = model_pre.module.features[15]
#     model.module.features[17].load(model_pre.module.features[17])
#     model.module.features[18] = model_pre.module.features[18]
#     model.module.features[21].load(model_pre.module.features[21])
#     model.module.features[22] = model_pre.module.features[22]
#     model.module.features[24].load(model_pre.module.features[24])
#     model.module.features[25] = model_pre.module.features[25]
#     model.module.features[28].load(model_pre.module.features[28])
#     model.module.features[29] = model_pre.module.features[29]
#     model.module.features[31].load(model_pre.module.features[31])
#     model.module.features[32] = model_pre.module.features[32]
#     model.module.classifier = model_pre.module.classifier
# else:
#     print("=> no checkpoint found at '{}'".format('checkpoint/Vgg13/0/model_best.pth.tar'))

# # Vgg16
# if os.path.isfile('checkpoint/Vgg16/0/model_best.pth.tar'):
#     print('=> loading checkpoint "{}"'.format('checkpoint/Vgg16/0/model_best.pth.tar'))
#     checkpoint = torch.load('checkpoint/Vgg16/0/model_best.pth.tar')
#     model_pre.load_state_dict(checkpoint['state_dict'])
#     model.module.features[0].load(model_pre.module.features[0])
#     model.module.features[1] = model_pre.module.features[1]
#     model.module.features[3].load(model_pre.module.features[3])
#     model.module.features[4] = model_pre.module.features[4]
#     model.module.features[7].load(model_pre.module.features[7])
#     model.module.features[8] = model_pre.module.features[8]
#     model.module.features[10].load(model_pre.module.features[10])
#     model.module.features[11] = model_pre.module.features[11]
#     model.module.features[14].load(model_pre.module.features[14])
#     model.module.features[15] = model_pre.module.features[15]
#     model.module.features[17].load(model_pre.module.features[17])
#     model.module.features[18] = model_pre.module.features[18]
#     model.module.features[20].load(model_pre.module.features[20])
#     model.module.features[21] = model_pre.module.features[21]
#     model.module.features[24].load(model_pre.module.features[24])
#     model.module.features[25] = model_pre.module.features[25]
#     model.module.features[27].load(model_pre.module.features[27])
#     model.module.features[28] = model_pre.module.features[28]
#     model.module.features[30].load(model_pre.module.features[30])
#     model.module.features[31] = model_pre.module.features[31]
#     model.module.features[34].load(model_pre.module.features[34])
#     model.module.features[35] = model_pre.module.features[35]
#     model.module.features[37].load(model_pre.module.features[37])
#     model.module.features[38] = model_pre.module.features[38]
#     model.module.features[40].load(model_pre.module.features[40])
#     model.module.features[41] = model_pre.module.features[41]
#     model.module.classifier = model_pre.module.classifier
# else:
#     print("=> no checkpoint found at '{}'".format('checkpoint/Vgg16/0/model_best.pth.tar'))

# Vgg16
# if os.path.isfile('checkpoint/Vgg19/0/model_best.pth.tar'):
#     print('=> loading checkpoint "{}"'.format('checkpoint/Vgg19/0/model_best.pth.tar'))
#     checkpoint = torch.load('checkpoint/Vgg19/0/model_best.pth.tar')
#     model_pre.load_state_dict(checkpoint['state_dict'])
#     model.module.features[0].load(model_pre.module.features[0])
#     model.module.features[1] = model_pre.module.features[1]
#     model.module.features[3].load(model_pre.module.features[3])
#     model.module.features[4] = model_pre.module.features[4]
#     model.module.features[7].load(model_pre.module.features[7])
#     model.module.features[8] = model_pre.module.features[8]
#     model.module.features[10].load(model_pre.module.features[10])
#     model.module.features[11] = model_pre.module.features[11]
#     model.module.features[14].load(model_pre.module.features[14])
#     model.module.features[15] = model_pre.module.features[15]
#     model.module.features[17].load(model_pre.module.features[17])
#     model.module.features[18] = model_pre.module.features[18]
#     model.module.features[20].load(model_pre.module.features[20])
#     model.module.features[21] = model_pre.module.features[21]
#     model.module.features[23].load(model_pre.module.features[23])
#     model.module.features[24] = model_pre.module.features[24]
#     model.module.features[27].load(model_pre.module.features[27])
#     model.module.features[28] = model_pre.module.features[28]
#     model.module.features[30].load(model_pre.module.features[30])
#     model.module.features[31] = model_pre.module.features[31]
#     model.module.features[33].load(model_pre.module.features[33])
#     model.module.features[34] = model_pre.module.features[34]
#     model.module.features[36].load(model_pre.module.features[36])
#     model.module.features[37] = model_pre.module.features[37]
#     model.module.features[40].load(model_pre.module.features[40])
#     model.module.features[41] = model_pre.module.features[41]
#     model.module.features[43].load(model_pre.module.features[43])
#     model.module.features[44] = model_pre.module.features[44]
#     model.module.features[46].load(model_pre.module.features[46])
#     model.module.features[47] = model_pre.module.features[47]
#     model.module.features[49].load(model_pre.module.features[49])
#     model.module.features[50] = model_pre.module.features[50]
#     model.module.classifier = model_pre.module.classifier
# else:
#     print("=> no checkpoint found at '{}'".format('checkpoint/Vgg19/0/model_best.pth.tar'))

best_prec = 0
min_loss = 1

data=open("temp.txt",'w') 
for name, param in model.named_parameters():
    print(name, param, file=data)
for name, param in model_pre.named_parameters():
    print(name, param, file=data)
data.close()

for epoch in range(0, 100):
    if epoch < 40:
        lr = 0.01
    elif epoch < 80:
        lr = 0.001
    else:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    train_total = 0
    train_correct = 0
    train_loss = 0
    # train for one epoch
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.long().cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, '-->grad_value:',parms.grad)

        train_loss += loss.item()
        ave_loss = train_loss/(i+1)

        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

        prec = train_correct / train_total
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, 100, i+1, len(train_loader), ave_loss, prec*100))

    # evaluate on test set
    # switch to evaluate mode
    model.eval()
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        total_loss = 0
        for i, (input, target) in enumerate(valid_loader):
            input, target = input.cuda(), target.long().cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)
            valid_total += target.size(0)
            valid_correct += (predicted == target).sum().item()

            total_loss += loss
    prec = valid_correct / valid_total
    ave_loss = total_loss/len(valid_loader)
    print('Accuary on test images:{:.2f}%'.format(prec*100))

    best_prec = max(prec, best_prec)
    min_loss = min(ave_loss, min_loss)

print('Best accuracy is: {:.2f}%, Minimum loss is: {:.4f}'.format(best_prec*100, min_loss))

data=open("weights2.txt",'w') 
for name, param in model.named_parameters():
    print(name, param, file=data)
for name, param in model_pre.named_parameters():
    print(name, param, file=data)
data.close()
