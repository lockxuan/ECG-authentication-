import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import numpy as np
from scipy.fftpack import fft,ifft
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms

per_data_length = 512
BATCH_SIZE = 64

def draw(label, predict):
	confmat = confusion_matrix(y_true=label, y_pred=predict)
	fig, ax = plt.subplots(figsize=(10.0, 10.0))
	ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
	for i in range(confmat.shape[0]):
		for j in range(confmat.shape[1]):
			ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
	plt.xlabel('predicted label')
	plt.ylabel('true label')
	plt.savefig('pic.png')
	plt.show()



def Loader(X_train, Y_train, X_test, Y_test):
	train_dataset = MyDataset(data=X_train,target=Y_train)
	train_loader = Data.DataLoader(
		dataset=train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
	)
	test_dataset = MyDataset(data=X_test,target=Y_test)
	test_loader = Data.DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=2,
	)

	return train_loader, test_loader


def readData(train_file, test_file, _fft = False):
	train_data = pd.read_csv(train_file, error_bad_lines=False)
	test_data = pd.read_csv(test_file, error_bad_lines=False)

	train_data = np.array(train_data, dtype=np.float64)
	test_data = np.array(test_data, dtype=np.float64)

	train_ecg, train_id, test_ecg, test_id = [], [], [], []

	if _fft:
		for i in range(len(train_data)):
			train_id.append(int(train_data[i][0]))
			train_ecg.append([[abs(fft(train_data[i][1:]))[0:int(per_data_length/2)]]])

		for i in range(len(test_data)):
			test_id.append(int(test_data[i][0]))
			test_ecg.append([[abs(fft(test_data[i][1:]))[0:int(per_data_length/2)]]])

	else:
		for i in range(len(train_data)):
			train_id.append(int(train_data[i][0]))
			train_ecg.append([train_data[i][1:]])
		for i in range(len(test_data)):
			test_id.append(int(test_data[i][0]))
			test_ecg.append([test_data[i][1:]])

	X_train = np.array(train_ecg)
	X_test = np.array(test_ecg)
	Y_train = np.array(train_id)
	Y_test = np.array(test_id)

	if _fft:
		X_train = X_train.reshape(X_train.shape[0], 1, per_data_length//2, 1)
		X_test = X_test.reshape(X_test.shape[0], 1, per_data_length//2, 1)
	else:
		X_train = X_train.reshape(X_train.shape[0], 1, per_data_length, 1)
		X_test = X_test.reshape(X_test.shape[0], 1, per_data_length, 1)

	return X_train, Y_train, X_test, Y_test

class MyDataset(Dataset):
	def __init__(self,data,target):
		super(MyDataset,self).__init__()
		self.data=data
		self.target=target

	def __getitem__(self,index):
		x=self.data[index]
		y=self.target[index]
		x=transforms.ToTensor()(x)

		return x[0].float(),y
	def __len__(self):
		return len(self.data)
