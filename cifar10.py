import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model.quadraticnet import AlexNet
from model.quadraticnet import ResNet18
import argparse

import os
import shutil
from tensorboardX import SummaryWriter

device = torch.device('cuda')

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--gpu-id', nargs='+', type=int, help='available GPU IDs')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='test model on test set')

args = parser.parse_args()

checkpoint_path = 'checkpoint'
summary_path = 'summary'

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# CIFAR10 dataset 
train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_transform)

valid_dataset, test_dataset = torch.utils.data.random_split(test_dataset, (int(0.5*len(test_dataset)), int(0.5*len(test_dataset))))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size)


# model = LinearNeuralNet(input_size, 5, num_classes).to(device)
model = AlexNet()
# model = ResNet18()
model = nn.DataParallel(model, device_ids=args.gpu_id).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
cudnn.benchmark = True

best_prec = 0
if args.resume:
    if os.path.isfile(args.resume):
        print('=> loading checkpoint "{}"'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {} best_acc {})".format(args.resume, checkpoint['epoch'], best_prec))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

writer = SummaryWriter(summary_path)

for epoch in range(args.start_epoch, args.epochs):
    if epoch < 150:
        lr = args.lr
    elif epoch < 225:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

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
        
        train_loss += loss.item()
        ave_loss = train_loss/(i+1)

        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

        prec = train_correct / train_total
        if i % args.print_freq == 0:
            writer.add_scalar('data/loss', ave_loss, epoch * len(train_loader) + i)
            writer.add_scalar('data/prec', prec, epoch * len(train_loader) + i)
            print('Epoch [{}/{}], Step [{}/{}], \
                Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, args.epochs, i+1, len(train_loader), ave_loss, prec*100))


    # evaluate on test set
    # switch to evaluate mode
    model.eval()
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            input, target = input.cuda(), target.long().cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)
            valid_total += target.size(0)
            valid_correct += (predicted == target).sum().item()
    prec = valid_correct / valid_total
    print('Accuary on test images:{:.2f}%'.format(prec*100))
    writer.add_scalar('data/accuracy', prec, epoch)

    is_best = prec > best_prec
    best_prec = max(prec, best_prec)

    filepath = os.path.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_prec,
            'optimizer': optimizer.state_dict(),
        }, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_path, 'model_best.pth.tar'))




