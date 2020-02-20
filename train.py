#!/usr/bin/env python
#
#	d-vector training using Keras.
#
#	July 2019

import numpy as np
import os, struct, keras
from tqdm import tqdm

from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

from utils.utils import *

# config
class Params:
	mel_dim		= 128			# mel spectrogram dimension
	num_speakers	= 77			# number of speakers (output dimension)
	batch_size	= 128
	num_epochs	= 50
	timesteps	= 1			# for stateful LSTM
	nval 		= 10
	lstm_stateful	= True 			# whether to use lstm stateful or not
	scpfile		= 'scp/data_ptBR.scp'
	model_dir 	= 'models'
	datadir		= 'data_ptBR'
	modelid		= 2			# model id
	if modelid == 2:
		lstm_stateful = False
params = Params()


def main():

	mel_dim		= params.mel_dim
	num_speakers	= params.num_speakers
	batch_size	= params.batch_size
	num_epochs	= params.num_epochs
	timesteps	= params.timesteps
	nval 		= params.nval
	scpfile		= params.scpfile
	model_dir 	= params.model_dir
	datadir		= params.datadir
	modelid		= params.modelid
	lstm_stateful	= params.lstm_stateful
	
	# output model directory
	try:
	    os.stat(model_dir)
	except:
	    os.mkdir(model_dir)

	# Load training data
	print('Reading training data...')
	x_train, y_train, num_train_examples = read_data(params, norm_type='-1_to_1')
	print(' %d examples have been loaded' % num_train_examples)

	# prepare training data
	x_train, y_train, x_val, y_val = prepare_data(x_train, y_train, params)
	num_train_examples = x_train.shape[0]
	num_val_examples = x_val.shape[0]
	assert num_train_examples == y_train.shape[0]
	assert num_val_examples == y_val.shape[0]
	print(' %d examples will effectively be used for training' % num_train_examples)
	print(' %d examples will be used for validation' % num_val_examples)

	# build model
	model = build_model(params)
	model.summary()

	# define optimization criterion and method: use these but feel at easy to try different ones
	keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])

	# Train/fit the model, iterating on the data in batches
	print('Start training...\n')
	filepath = model_dir + '/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	if lstm_stateful is False:
		# no lstm stateful case
		model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2)
	else:
		# lstm stateful
		for i in range(num_epochs):
			model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=False, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2)
			model.reset_states()

	# check on the model performance
	print('\n\nSUMMARY OF THE PERFORMANCE OF THE MODEL:')
	scores = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=8)
	print("Model Accuracy: %.2f%%\n\n" % (scores[1]*100))


if __name__ == "__main__":
	main()

