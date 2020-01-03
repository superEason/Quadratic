import os
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from constants import *
from PIL import Image

class CifarDataset(Dataset):
	def __init__(self, data_type, image_size=32):  # 0 - train 1 - val 2 - test

		# print('=> loading cifar10 data...')

		normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

		self.data_type = data_type

		if self.data_type == 0:
			self.data_path = os.path.join(train_test_split_path, 'train.npz')
			self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
		elif self.data_type == 1:
			self.data_path = os.path.join(train_test_split_path, 'val.npz')
			self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
		else:
			self.data_path = os.path.join(train_test_split_path, 'test.npz')
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])

		self.data = np.load(self.data_path)
		self.input = self.data['input']
		if self.data_type != 2:
			self.target = self.data['target']
		else:
			self.id = self.data['id']
		self.len = len(self.input)

		set_map = {0: 'train', 1: 'val', 2: 'test'}
		print('{set_type} Set: {length}'.format(set_type=set_map[data_type], length=self.len))

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		input_item = self.input[idx]
		if self.data_type != 2:
			target_item = self.target[idx]
		else:
			id_item = self.id[idx]

		img = Image.fromarray(np.transpose(input_item, (1, 2, 0)))
		input_item = self.transform(img)

		if self.data_type != 2:
			return input_item, target_item
		else:
			return id_item, input_item

def load_data(batch_size, data_type):
	training = True if data_type == 0 else False

	cifar_dataset = CifarDataset(data_type)
	Dataloader =  DataLoader(cifar_dataset, batch_size=batch_size, shuffle=training, drop_last=training, num_workers=8)

	return Dataloader