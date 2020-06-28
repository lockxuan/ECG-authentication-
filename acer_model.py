import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import numpy as np
from scipy.fftpack import fft,ifft
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torchvision.models as models

#hyper parameters
per_data_length = 512
numOfPeople = 12

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

class Channel(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(Channel, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.max_pool = nn.AdaptiveMaxPool1d(1)

		self.MLP = nn.Sequential(
			#nn.Conv1d(in_planes, in_planes//ratio, 1, bias = False),
			nn.Linear(in_planes, in_planes//ratio),
			nn.ReLU(),
			nn.Linear(in_planes//ratio, in_planes))
			#nn.Conv1d(in_planes//ratio, in_planes, 1, bias = False))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = self.avg_pool(x)
		max_out = self.max_pool(x)
		avg_out = torch.reshape(avg_out, (avg_out.size()[0],-1))
		max_out = torch.reshape(max_out, (max_out.size()[0],-1))

		avg_out = self.MLP(avg_out).unsqueeze(2)
		max_out = self.MLP(max_out).unsqueeze(2)

		out = self.sigmoid(avg_out + max_out)

		return out * x

class Spatial(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        #x = self.sigmoid(x)
        return out * x

class ECGmodel(nn.Module):
	def __init__(self, _fft=False):
		super(ECGmodel, self).__init__()
		self.fft = _fft

		self.conv1 = nn.Sequential(
			#nn.BatchNorm1d(1),
			nn.Conv1d(1,32,3, 1, 1),
			nn.ReLU(),
			nn.Conv1d(32,64,3, 1, 1),
			#nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.MaxPool1d(kernel_size=2),
			#nn.ReLU(),
			nn.Conv1d(64,128,3, 1, 1),
			#nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=4),
			#nn.ReLU()
			)


		self.out = nn.Sequential(

			nn.Linear(4096 if self.fft else 8192,1024),
			nn.ReLU(),
			#nn.Linear(1024,512),
			#nn.Tanh(),
			nn.Linear(1024,128),
			nn.Tanh(),
			nn.Linear(128,numOfPeople),
			)
		"""
		self.out2 = nn.Sequential(
			nn.Linear(256,128),
			nn.Tanh(),
			nn.Linear(128,numOfPeople),
			)
		"""


	def forward(self, x):
		#x = self.norm(x)
		#x = self.norm2(x)

		x = self.conv1(x)
		#a = self.conv2(x)
		#print(x.size()) batch 64 / Width 25 / Channel 128 / (h=1)
		#m = self.max_pool(x)
		#a = self.avg_pool(x)

		#x = torch.cat((m, a), dim=1)



		x = torch.reshape(x, (x.size()[0],-1))

		output = self.out(x)
		#print(output.shape)
		
		return output

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
								stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)
		self.conv3 = nn.Conv1d(planes, self.expansion *
								planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm1d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion*planes,
						kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm1d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv1d(
			in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(planes)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
								stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv1d(in_planes, self.expansion*planes,
							kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm1d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out



class ResNet(nn.Module):
	def __init__(self, block, num_blocks, _fft=False):
		super(ResNet, self).__init__()
		self.fft = _fft
		self.in_planes = 64

		self.conv1 = nn.Sequential(
			nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
			)

		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[0], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[0], stride=2)
		self.linear = nn.Linear(6144, 128)
		self.linear2 = nn.Linear(128, numOfPeople)

		self.out = nn.Sequential(

			nn.Linear(3200 if self.fft else 6144,1024),
			nn.ReLU(),
			#nn.Linear(1024,512),
			#nn.ReLU(),
			nn.Linear(1024,128),
			nn.Tanh(),
			nn.Linear(128,numOfPeople),
			)

		self.drop = nn.Dropout(0.5)


	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)


	def forward(self, x):

		out = self.conv1(x)
		out = self.layer1(out)
		out = self.drop(out)
		out = self.layer2(out)
		out = self.drop(out)
		out = self.layer3(out)
		out = self.drop(out)
		#out = self.layer4(out)
		#out = self.drop(out)
		#print(out.shape)
		out = F.avg_pool1d(out, 4)
		#print(out.shape)
		out = out.view(out.size(0), -1)
		out = F.tanh(self.linear(out))
		out = self.linear2(out)
		return out


class pretrainedResNet(nn.Module):
	"""docstring for pretrainedResNet"""
	def __init__(self):
		super(pretrainedResNet, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)

		self.fc = nn.Sequential(
			nn.Linear(1000, 128),
			#nn.Tanh(),
			nn.Linear(128, numOfPeople)
			)

	def forward(self, x):
		x = torch.cat((x, x, x),dim=1)
		x = x.unsqueeze(-1)

		x = self.resnet18(x)
		#print(x.shape)
		x = self.fc(x)

		return x
		


class NewModel(nn.Module):
	def __init__(self):
		super(NewModel, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=3 ,padding=1, stride=1),
			nn.ReLU(),
			nn.Conv1d(32, 32, kernel_size=3 ,padding=1, stride=1),
			#nn.MaxPool1d(kernel_size=2),
			#nn.Dropout(0.5),
			nn.ReLU()
			)

		self.conv2 = nn.Sequential(
			nn.Conv1d(32, 64, kernel_size=3 ,padding=1, stride=1),
			nn.ReLU(),
			
			nn.Conv1d(64, 64, kernel_size=3 ,padding=1, stride=1),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.5),	
			#nn.ReLU()
			)

		self.conv3 = nn.Sequential(
			nn.Conv1d(64, 128, kernel_size=3 ,padding=1, stride=1),
			#nn.Dropout(0.5),
			nn.ReLU(),
			nn.Conv1d(128, 128, kernel_size=3 ,padding=1, stride=1),
			nn.MaxPool1d(kernel_size=4),
			nn.Flatten()
			)

		self.fc = nn.Sequential(
			nn.Linear(8192, 1024),
			nn.ReLU(),
			#nn.Linear(1024,512),
			#nn.Tanh(),
			nn.Linear(1024,128),
			nn.Tanh(),
			nn.Linear(128, numOfPeople)
			)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		
		x = self.fc(x)

		return x

class LSTMModel(nn.Module):
	def __init__(self):
		super(LSTMModel, self).__init__()

		self.lstm = nn.LSTM(1,16, batch_first=True, bidirectional=False)

		self.fc = nn.Sequential(
			#nn.Linear(51200, 6400),
			#nn.Dropout(0.5),
			#nn.ReLU(),
			nn.Linear(8192, 1024),
			nn.ReLU(),
			nn.Linear(1024, 128),
			#nn.Tanh(),
			nn.Linear(128, 41),
			#nn.Tanh(),
			#nn.Linear(128, 41),
			#nn.ReLU(),

			)
		

	def forward(self, x):
		x = x.transpose(2,1)
		x , (h,c) = self.lstm(x)
		#x = F.tanh(x)

		x = nn.Flatten()(x)
		#x = x.squeeze(-1)
		#print(x.shape)


		
		x = self.fc(x)
		return x

class TinyLSTMModel(nn.Module):
	def __init__(self):
		super(TinyLSTMModel, self).__init__()
		
		self.conv1 = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=3 ,padding=1, stride=1),
			nn.ReLU(),
			#nn.BatchNorm1d(32),
			nn.Conv1d(32, 64, kernel_size=3 ,padding=1, stride=1),
			nn.MaxPool1d(kernel_size=4),
			nn.Dropout(0.5),
			nn.Tanh(),
			)


		self.fc = nn.Sequential(
			nn.Linear(8192, 1024),
			#nn.Dropout(0.5),
			nn.Tanh(),
			nn.Linear(1024, 41),
			#nn.Tanh(),
			#nn.Linear(128, 41),

			)
		self.lstm = nn.LSTM(64,64, batch_first=True, bidirectional=False)

	def forward(self, x):
		x = self.conv1(x)
		x = x.transpose(2,1)
		x , (h,c) = self.lstm(x)
		#x = F.tanh(x)


		#x = nn.Flatten()(att)
		x = nn.Flatten()(x)
		#x = x.squeeze(-1)
		
		x = self.fc(x)

		return x
		

class TinyModel(nn.Module):
	def __init__(self):
		super(TinyModel, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=3 ,padding=1, stride=1),
			nn.ReLU(),
			#nn.BatchNorm1d(32),
			nn.Conv1d(32, 64, kernel_size=3 ,padding=1, stride=1),
			#nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.5),
			nn.Tanh(),
			)
		self.conv2 = nn.Sequential(
			nn.Conv1d(128, 128, kernel_size=7 ,padding=3, stride=2),
			nn.ReLU()
			#nn.MaxPool1d(kernel_size=4),
			#nn.Flatten()
			)

		self.att = nn.Sequential(
			nn.Conv1d(64, 128, 3, padding=1, stride=2),
			nn.BatchNorm1d(128),
			nn.Tanh(),
			nn.Conv1d(128, 128, 3, padding=1, stride=1),
			#nn.Tanh()
			#nn.AvgPool1d(kernel_size=4),
			)
		self.max_pool = nn.MaxPool1d(kernel_size=4)
		self.avg_pool = nn.AvgPool1d(kernel_size=4)
		self.ad_pool = nn.AdaptiveMaxPool1d(2)
		self.fc2 = nn.Sequential(
			nn.Linear(128*2,128),
			nn.Tanh(),
			nn.Linear(128,41),
			)


	def forward(self, x):
		x = self.conv1(x)
		x1 = self.avg_pool(x)
		x2 = self.max_pool(x)
		x1 = self.att(x1)
		x2 = self.att(x2)

		#x2+= self.att(x)
		#x = nn.Flatten()(x)
		#x = self.ad_pool(x1+x2)
		x = x1+x2
		x = self.conv2(x)
		x = self.ad_pool(x)
		#x = self.ad_pool(x1+x2)

		x = nn.Flatten()(x)
		#x = x.squeeze(-1)


		
		x = self.fc2(x)

		return x

