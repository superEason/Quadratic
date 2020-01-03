import os
import pandas as pd
import numpy as np
from constants import *


image_size = 32
split_portion = 0.9


def train_val_set():
	data_path = os.path.join(train_test_split_path, 'train.csv')
	data = pd.read_csv(data_path)
	length = len(data)
	input = np.asarray(data.iloc[:, 1:-1], dtype=np.uint8).reshape(length, 3, image_size, image_size)   # !!!
	target = np.asarray(data.iloc[:, -1], dtype=np.uint8)

	index = np.arange(length)
	np.random.shuffle(index)

	split_point = int(length * split_portion)
	train_index = index[:split_point]
	val_index = index[split_point:]

	np.savez(os.path.join(train_test_split_path, 'train.npz'), input=input[train_index], target=target[train_index])
	np.savez(os.path.join(train_test_split_path, 'val.npz'), input=input[val_index], target=target[val_index])

def test_set():
	data_path = os.path.join(train_test_split_path, 'test.csv')
	data = pd.read_csv(data_path)
	length = len(data)
	input = np.asarray(data.iloc[:, 1:], dtype=np.uint8).reshape(length, 3, image_size, image_size)
	id = np.asarray(data.iloc[:, 0])

	np.savez(os.path.join(train_test_split_path, 'test.npz'), id=id, input=input)

if __name__ == '__main__':
	train_val_set()
	test_set()

