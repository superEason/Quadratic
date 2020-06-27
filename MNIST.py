import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

import model.simplenet as simplenet

device = torch.device('cuda')

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
valid_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64)

model = simplenet.SimpleNet_6()
model = nn.DataParallel(model, device_ids=[0]).cuda()
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

best_prec = 0
min_loss = 1

for epoch in range(0, 20):
    if epoch < 10:
        lr = 0.01
    elif epoch < 15:
        lr = 0.001
    # else:
        # lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    train_total = 0
    train_correct = 0
    train_loss = 0
    # train for one epoch
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
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
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, 20, i+1, len(train_loader), ave_loss, prec*100))

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
