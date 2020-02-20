#!/usr/bin/env python
#
#
#	d-vector estimation
#	July 2019

import os, sys, argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

from keras.models import Sequential, load_model
from keras import backend as K

from utils.utils import *

# config
class Params:
	mel_dim		= 128			# mel spectrogram dimension
	num_speakers	= 77			# number of speakers (output dimension)
	batch_size	= 128
	timesteps	= 1			# for stateful LSTM
	lstm_stateful	= False 			# whether to use lstm stateful or not
	scpfile		= 'scp/data_ptBR.scp'
	datadir		= 'data_ptBR'
	modelid		= 2			# model id
	bottleneck_size = 64			# bottleneck layer size
	if modelid == 2:
		lstm_stateful = False
params = Params()

# d-vector as the framewise mean of the bottleneck layer
def get_dvector(model2, x_train, params):

	batch_size = params.batch_size
	mel_dim = params.mel_dim
	embed_dim = params.bottleneck_size

	D = []
	for x in tqdm(x_train):
		x = x.reshape([1,batch_size,mel_dim])
		bno = model2([x])[0]
		aux = np.reshape(bno,[batch_size,embed_dim])
		D += aux.tolist()
	D = np.asarray(D)
	dvector = np.mean(D,axis=0)
	return (D, dvector)


# d-vector as the framewise PCA of the bottleneck layer
def get_dvector_pca(model2, x_train, params):

	batch_size = params.batch_size
	mel_dim = params.mel_dim
	embed_dim = params.bottleneck_size

	D = []
	for x in tqdm(x_train):
		x = x.reshape([1,batch_size,mel_dim])
		bno = model2([x])[0]
		aux = np.reshape(bno,[batch_size,embed_dim])
		D += aux.tolist()
	D = np.asarray(D)
	pca = PCA()
	pca.fit(D)
	dvector = np.squeeze(pca.components_[0,:])
	return (D, dvector)


def main():

	# input argument processing
	parser = argparse.ArgumentParser()
	parser.add_argument('--speakerid', default='0', help='Speaker ID (default=\'0\')', required=True)
	parser.add_argument('--out_dvector_file', help='Output filen name (default=\'dvector_speakerid.npy\').')
	parser.add_argument('--model_path', required=True, help='Path to a trained model.')
	parser.add_argument('--mode', default='mean', help='Method to calculate d-vector: mean or PCA (default=\'mean\').')
	args = parser.parse_args()
	speakerid = args.speakerid
	out_dvector_file = args.out_dvector_file
	dvector_method = args.mode
	model_path = args.model_path
	print('\nSpeaker id: %s' %speakerid)
	print('Output d-vector file: %s' %out_dvector_file)
	print('D-vector calculation method: %s' %dvector_method)

	# configs
	mel_dim		= params.mel_dim
	num_speakers	= params.num_speakers
	batch_size	= params.batch_size
	timesteps	= params.timesteps
	lstm_stateful	= params.lstm_stateful
	scpfile		= params.scpfile
	datadir		= params.datadir
	modelid		= params.modelid

	# Load speaker id data
	print('Reading data...')
	x_train, num_train_examples = read_data_speakerid(int(speakerid), params, norm_type='-1_to_1')
	print(' %d examples have been loaded' % num_train_examples)

	# model and layers
	print('Building model...')
	model1 = build_model(params)
	if modelid == 1 or modelid == 2:
		model2 = K.function([model1.layers[0].input],[model1.layers[5].output])
	else:
		model2 = K.function([model1.layers[0].input],[model1.layers[4].output])
	model1.summary()

	# load weights
	print('Loading weights...')
	model1.load_weights(model_path)

	# reshape training data
	x_train = prepare_data_inference(x_train, params)
	num_train_examples = x_train.shape[0]
	print(' %d examples will effectively be used for inference' % num_train_examples)

	# get bottleneck output
	print('Start d-vector estimation...\n')
	if dvector_method == 'pca':
		[D, dvector] = get_dvector_pca(model2, x_train, params)
	elif dvector_method == 'mean':
		[D, dvector] = get_dvector(model2, x_train, params)
	else:
		raise ValueError("Unkown d-vector calculation method.\n")

	print('d-vector:\n')
	print(dvector)
	print('Dimension of the D matrix : {}'.format(D.shape))
	print('Dimension of the d vector : {}'.format(dvector.shape))

	# save dvector
	np.save(out_dvector_file, dvector)

if __name__ == "__main__":
	main()

