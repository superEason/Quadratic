import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model.quadraticnet import AlexNet
from model.quadraticnet import ResNet18
import argparse


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

        if i % args.print_freq == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), ave_loss))


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

    print('Accuracy of the network on the {} test images: {} %'.format(len(valid_loader), 100 * valid_correct / valid_total))





