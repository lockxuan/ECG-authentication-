import numpy as np
import csv
import sys
from biosppy.signals import ecg
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks
import statistics
import random
from argparse import ArgumentParser
from scipy.signal import freqz
from scipy.signal import butter, lfilter


PERWAVE = 400
LRpoints = 200
WAVENUM = 1

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y



def outputfile(fileName, ans):
	try:
		f = open(fileName+'.csv',"w")
		w = csv.writer(f)
		title = []
		title.append('id')
		for n in range(PERWAVE * WAVENUM):
			title.append(n+1)
		w.writerow(title)
		for i in range(len(ans)):
			data_detail=[]
			for tempp in ans[i]:
				data_detail.append(tempp)
			w.writerow(data_detail) 
		f.close()
		print('success')
	except:
		f.close()
		print('fail')

def get_data(fileName):
	ecg_data = []
	user_id = []
	with open(fileName) as datafile:
		csvReader = csv.reader(datafile)
		for row in csvReader:
			if row[0] == "id":
				continue
			temp=[]
			for i in row[1:]:
				temp.append(float(i))
			ecg_data.append(temp)
			user_id.append(round(float(row[0])))

	return user_id, ecg_data

def seperate(user, ecg_data):
	"""
	隨機取 2 (k=2)筆 data，為了 balance data
	"""
	fs = 500.0
	lowcut = 1.0
	highcut = 15.0


	total = set(user)
	count = 0
	_id = []
	_ecg = []
	USER = 0
	for i in total:
		temp = user.count(i)
		if temp>2:
			temp_ecg = random.sample(ecg_data[count : count+temp], k=3)

			for data in temp_ecg:
				_id.append(USER)
				data = butter_bandpass_filter(data, lowcut, highcut, fs, order=5)
				_ecg.append(data)
			USER += 1
		count += temp

	return _id, _ecg


def resample(user, ecg_data, output):
	"""
	用peak to peak，resample方法，取兩peak間所有點再resample到所需點 (PERWAVE)
	"""
	r_peak = []
	for n in range(len(user)):
		for i in range(len(ecg_data[n])):
			ecg_data[n][i]=float(ecg_data[n][i])

		peak = find_peaks(ecg_data[n], height=0, distance = 300)

		#print('no :',n,' / id :',user[n], ' / peak_num :',len(peak[0]),' / mean :',statistics.mean(peak[1]['peak_heights']),' / var. :',statistics.variance(peak[1]['peak_heights'] ))
		r_peak.append(peak[0])

	count = 0
	train_data = []
	test_data = []

	USER = 0
	for i in range(len(set(user))):
		num = -1
		id_count = user.count(i)
		temp_id = user[count : count+id_count]
		temp_ecg = ecg_data[count : count+id_count]
		data = []
		for j in range(id_count):
			print('user', i, ' no', j, len(r_peak[count+j]))

			sample_data = []
			k=0
			while r_peak[count+j][k]!=r_peak[count+j][-WAVENUM] or k==0:
				temp=ecg_data[count+j][ r_peak[count+j][k] : r_peak[count+j][k+1] ]
				temp=np.array(temp)
				temp=signal.resample(temp, PERWAVE * 1)


				k+=1
				num+=1
				data.append(temp)
				data[num]=np.insert(data[num], 0, USER)

		count += id_count

		#random 取 test 20 個波
		random.shuffle(data)
		test_data += data[:20]
		train_data += data[20:]

		print('user',USER,'ok')

		USER += 1

	outputfile(output + '_train', train_data)
	outputfile(output + '_test', test_data)

def normal(user, ecg_data, output):
	"""
	從peak左右個取 LRpoints 個點
	"""
	

	r_peak = []
	for n in range(len(user)):
		for i in range(len(ecg_data[n])):
			ecg_data[n][i]=float(ecg_data[n][i])

		#peak = find_peaks(ecg_data[n], height=0, distance = 300)
		#peak = ecg.engzee_segmenter(signal=ecg_data[n], sampling_rate=500.0, threshold=0.5)
		peak = ecg.christov_segmenter(signal=ecg_data[n], sampling_rate=500.0)
		#print('no :',n,' / id :',user[n], ' / peak_num :',len(peak[0]),' / mean :',statistics.mean(peak[1]['peak_heights']),' / var. :',statistics.variance(peak[1]['peak_heights'] ))
		r_peak.append(peak[0])

	count = 0
	train_data = []
	test_data = []

	USER = 0
	for i in range(len(set(user))):


		num = -1
		id_count = user.count(i)

		temp_id = user[count : count+id_count]
		temp_ecg = ecg_data[count : count+id_count]
		data = []

		test_num = -1
		_test_data = []
		for j in range(id_count):
			print('user', USER, ' no', j, len(r_peak[count+j]))

			if j!=2:
				k=0
				total_len = len(ecg_data[count+j])
				peak_num = len(r_peak[count+j])
				if r_peak[count+j][k]-LRpoints<0:
					k+=1
				while k<peak_num:
					if r_peak[count+j][k]+LRpoints < total_len:
						temp=ecg_data[count+j][ r_peak[count+j][k]-LRpoints : r_peak[count+j][k]+LRpoints ]
						temp=np.array(temp)
						num+=1
						data.append(temp)
						data[num]=np.insert(data[num], 0, USER)
					k+=1
			else:
				k=0
				total_len = len(ecg_data[count+j])
				peak_num = len(r_peak[count+j])
				if r_peak[count+j][k]-LRpoints<0:
					k+=1
				while k<peak_num:
					if r_peak[count+j][k]+LRpoints < total_len:
						temp=ecg_data[count+j][ r_peak[count+j][k]-LRpoints : r_peak[count+j][k]+LRpoints ]
						temp=np.array(temp)
						test_num+=1
						_test_data.append(temp)
						_test_data[test_num]=np.insert(_test_data[test_num], 0, USER)
					k+=1


			#random 取 test 20 個波
		random.shuffle(data)
		#test_data += data[:20]
		#train_data += data[20:]
		test_data += _test_data
		train_data += data

		print('user',USER,'ok')
		USER += 1
		count += id_count

	outputfile(output + '_train', train_data)
	outputfile(output + '_test', test_data)


def _args():
	parser = ArgumentParser()
	parser.add_argument('--input_file')
	parser.add_argument('--output_file')
	parser.add_argument('--sep_type', default='normal')
	args = parser.parse_args()
	return args


def main():
	args = _args()

	user_id, ecg_data = get_data(args.input_file)
	_id, _ecg,= seperate(user_id, ecg_data)
	#if args.sep_type=='resample' or 'r' or 're':
	#	resample(_id, _ecg, args.output_file)
	#else:
	#	normal(_id, _ecg, args.output_file)
	normal(_id, _ecg, args.output_file)






if __name__ == '__main__':
	main()




