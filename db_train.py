import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
from scipy.fftpack import fft,ifft
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix, f1_score

from argparse import ArgumentParser

from db_model import threelayersCNN, ResNet, BasicBlock, Bottleneck, sixlayersCNN,ChannelAttModel,CNNLSTMModel,pretrainedResNet,LSTMModel
from db_utils import draw,Loader,readData

from tensorboardX import SummaryWriter
writer = SummaryWriter()

#hyper parameters
EPOCH = 251
BATCH_SIZE = 64
LR = 9e-4
per_data_length = 400
numOfPeople = 41

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def get_parameter_number(net):
	total_num = sum(p.numel() for p in net.parameters())
	trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}



def main(args):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	#cnn = threelayersCNN(_fft = args.fft).to(device)
	#cnn = sixlayersCNN().to(device)
	#cnn = CNNLSTMModel().to(device)
	cnn = ChannelAttModel().to(device)
	#cnn = LSTMModel().to(device)
	#cnn = pretrainedResNet().to(device)
	#cnn = ResNet(Bottleneck, [1, 1, 1, 1]).to(device)
	print(cnn)
	print(get_parameter_number(cnn))

	# load data
	X_train, Y_train, X_test, Y_test = readData(args.train_file, args.test_file, args.fft)
	train_loader, test_loader = Loader(X_train, Y_train, X_test, Y_test)

	# optimizer and loss function
	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-5, eps=1e-6)
	loss_func = nn.CrossEntropyLoss()


	best_pre = None
	best_acc = 0
	best_epoch = 0
	best_loss = [0 ,0]

	# training
	for epoch in range(EPOCH):

		train_loss = 0
		n=0
		for data in train_loader:
			_ecg, label=data

			_ecg = _ecg.to(device)
			label = label.to(device).long()

			optimizer.zero_grad()
			output = cnn(_ecg)
			loss = loss_func(output, label)
			train_loss+=loss.data
			loss.backward()
			optimizer.step()
			n+=1
				

		#testing
		total=0
		correct=0
		test_loss=0
		k=0
		pre = torch.tensor([0])
		lab = torch.tensor([0])
		with torch.no_grad():
			for data in test_loader:
				input, label=data

				test_x = Variable(input).to(device)
				test_y = Variable(label).to(device)
				test_output=cnn(test_x)

				predicted = torch.max(test_output, 1)[1].cpu()
				loss = loss_func(test_output, test_y)
				test_loss+=loss.data
				k+=1

				pre = torch.cat((pre, torch.max(test_output, 1)[1].cpu()), 0)
				lab = torch.cat((lab, label), 0)
				total += label.size(0)
				correct += (predicted == label).sum().item()
					
			train_loss /= n
			test_loss /= k
			accuracy_t = float((correct / total) * 100)
			if accuracy_t>best_acc:
				best_pre = pre
				best_acc = accuracy_t
				best_epoch = epoch
				best_loss[0] = train_loss
				best_loss[1] = test_loss


		writer.add_scalar('Train/Loss', train_loss.data, epoch)
		writer.add_scalar('Test/Loss', test_loss.data, epoch)
		writer.add_scalar('Test/accuracy', accuracy_t, epoch)
		print("Epoch:", epoch, '| train loss: %.4f' % train_loss.data, '| test loss: %.4f' % test_loss.data, '| test accuracy: %.2f' % accuracy_t)


	print()
	f1 = f1_score(lab[1:], best_pre[1:], average='weighted')*100
	print('best epoch:',best_epoch, ' | accuracy: ', best_acc,' | train loss: %.4f' % best_loss[0].data,' | test loss: %.4f' % best_loss[1].data, ' | f1: %.4f' % f1)
	draw(label=lab[1:], predict=best_pre[1:])



def _args():
	parser = ArgumentParser()
	parser.add_argument("--train_file")
	parser.add_argument('--test_file')
	parser.add_argument('--fft', default=False, type=bool)

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = _args()
	main(args)



