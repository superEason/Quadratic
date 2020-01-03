import argparse
import os
import time
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from dataloader import load_data
from densenet import *
from constants import *

from tensorboardX import SummaryWriter

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

best_prec = 0

def main():
	global args, best_prec
	args = parser.parse_args()

	print('=> Building model...')
	model = densenet(num_classes=10, depth=190, growthRate=40)

	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)

	model = nn.DataParallel(model, device_ids=args.gpu_id).cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	cudnn.benchmark = True

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

	print('=> loading cifar10 data...')
	trainloader = load_data(batch_size=args.batch_size, data_type=0)
	valloader = load_data(batch_size=args.batch_size, data_type=1)

	if args.evaluate:
		validate(valloader, model, criterion)
		return

	if args.test:
		testloader = load_data(batch_size=args.batch_size, data_type=2)
		test(testloader, model)
		return

	if not os.path.exists(summary_path):
		os.makedirs(summary_path)

	writer = SummaryWriter(summary_path)

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(trainloader, model, criterion, optimizer, writer, epoch)

		# evaluate on test set
		prec = validate(valloader, model, criterion)

		writer.add_scalar('data/accuracy', prec, epoch)

		# remember best precision and save checkpoint
		is_best = prec > best_prec
		best_prec = max(prec, best_prec)

		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_acc': best_prec,
			'optimizer': optimizer.state_dict(),
		}, is_best, checkpoint_path)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def train(trainloader, model, criterion, optimizer, writer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(trainloader):
		# measure data loading time
		data_time.update(time.time() - end)

		input, target = input.cuda(), target.long().cuda()

		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		prec = accuracy(output, target)[0]
		losses.update(loss.item(), input.size(0))
		top1.update(prec.item(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			writer.add_scalar('data/loss', loss, epoch * len(trainloader) + i)
			writer.add_scalar('data/prec', prec, epoch * len(trainloader) + i)

			print('Epoch: [{0}][{1}/{2}]\t'
			      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
			      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			      'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
				epoch, i, len(trainloader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			input, target = input.cuda(), target.long().cuda()

			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			prec = accuracy(output, target)[0]
			losses.update(loss.item(), input.size(0))
			top1.update(prec.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
				      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				      'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses,
					top1=top1))

	print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

	return top1.avg


def test(test_loader, model):
	prediction = np.zeros((10000, 2))

	model.eval()

	with torch.no_grad():
		for i, (id, input) in enumerate(test_loader):
			input = input.cuda()
			output = model(input)
			_, predicted = output.max(1)
			prediction[id, 0] = id
			prediction[id, 1] = predicted.cpu().numpy()

		prediction = prediction.astype(int)
		pd_data = pd.DataFrame(prediction, columns=['ID', 'Category'])
		pd_data.to_csv(os.path.join(train_test_split_path, 'prediction.csv'), index=False)


def save_checkpoint(state, is_best, fdir):
	filepath = os.path.join(fdir, 'checkpoint.pth.tar')
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
	if epoch < 150:
		lr = args.lr
	elif epoch < 225:
		lr = args.lr * 0.1
	else:
		lr = args.lr * 0.01

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


if __name__ == '__main__':
	main()
