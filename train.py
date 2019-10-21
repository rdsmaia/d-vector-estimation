#!/usr/bin/env python
#
#	A simple tool for speaker identification using Keras.
#
#	Training script.
#
#	May 2017
#
#	Ranniery Maia

import numpy as np
import keras, os, struct, argparse
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle

from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# config data
class Configs:
	feat_dim	= 128
	num_speakers	= 269
	batch_size	= 128
	num_epochs	= 100
	timesteps	= 1
	nval 		= 10	# percentage of validation samples
	lstm_stateful	= False
	datadir		= 'data'
	scpfile		= 'scp/train.scp'
	model_dir 	= 'checkpoints'
configs = Configs()

# function definition
def pause():
    programPause = input("Press the <ENTER> key to continue...")

def read_data(script_file, datadir, num_mels, num_speakers):
	if script_file is None or not os.path.isfile(script_file):
		raise IOError('No such file %s' % script_file)
	specfiles = []
	speaker_ids = []
	for files in open(script_file, 'r'):
		input_filename, speaker_id = files.split(' ', 1)
		specfiles.append(input_filename)
		speaker_ids.append(speaker_id)
	inputs  = np.empty((0, num_mels))
	outputs = np.empty((0, num_speakers))
	for i, files in enumerate(tqdm(specfiles)):
		mel = np.load(os.path.join('data',files))
		out = np.zeros([mel.shape[0],num_speakers])
		out[:,int(speaker_ids[i])] = 1.
		inputs = np.append(inputs, mel, axis=0)
		outputs = np.append(outputs, out, axis=0)

	mean_var_scaler = preprocessing.StandardScaler()
	inputs = mean_var_scaler.fit_transform(inputs)
	assert len(inputs) == len(outputs)

	num_examples = len(inputs)
	return inputs, outputs, num_examples

def read_data_v2(script_file, datadir, num_mels, num_speakers):
	if script_file is None or not os.path.isfile(script_file):
		raise IOError('No such file %s' % script_file)
	specfiles = []
	speaker_ids = []
	for files in open(script_file, 'r'):
		input_filename, speaker_id = files.split(' ', 1)
		specfiles.append(input_filename)
		speaker_ids.append(speaker_id)
	inputs  = []
	outputs = []
	for i, files in enumerate(tqdm(specfiles)):
		mel = np.load(os.path.join(datadir,files))
		out = np.zeros([mel.shape[0],1]) + int(speaker_ids[i])
		inputs += mel.tolist()
		outputs += out.tolist()
	inputs = np.asarray(inputs)
	outputs = np_utils.to_categorical(outputs, num_speakers)

	mean_var_scaler = preprocessing.StandardScaler()
	inputs = mean_var_scaler.fit_transform(inputs)

	assert len(inputs) == len(outputs)

	num_examples = len(inputs)
	return inputs, outputs, num_examples

def build_model(feat_dim, num_speakers, batch_size, timesteps, modelid=2, lstm_stateful=False):
	model = Sequential()

	# option 1
	if modelid == 1:
		model.add(TimeDistributed(Dense(512, activation='tanh'), batch_input_shape=(batch_size, timesteps, feat_dim)))
		model.add(TimeDistributed(Dense(1024, activation='tanh')))
		model.add(TimeDistributed(Dense(1024, activation='tanh')))
		model.add(TimeDistributed(Dense(1024, activation='tanh')))
		model.add(LSTM(512, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(Dense(512, activation='tanh'))
		model.add(Dense(num_speakers, activation='softmax'))
	# option 2
	elif modelid == 2:
		model.add(Dense(1024, activation='tanh', input_shape=(batch_size, feat_dim)))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(num_speakers, activation='softmax'))
	# option 3
	elif modelid == 3:
		model.add(LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', batch_input_shape=(batch_size, timesteps, feat_dim), stateful=True))
		model.add(LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(LSTM(512, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(Dense(512, activation='tanh'))
		model.add(Dense(512, activation='tanh'))
		model.add(Dense(num_speakers, activation='softmax'))
	else:
		print(f'Model {modelid} is not ready yet.')
		sys.exit()

        #model.add(LSTM(512, activation='tanh', recurrent_activation='sigmoid', stateful=True))

	return(model)

def prepare_data(x_train, y_train, timesteps, batch_size, nval, lstm_stateful=False):

	# shuffle data
	if lstm_stateful is False:
		x_train, y_train = shuffle(x_train, y_train, random_state=0)

	lab_dim = x_train.shape[1]
	cmp_dim = y_train.shape[1]

	num_train_examples = x_train.shape[0]
	num_batches = int(np.fix(num_train_examples/batch_size))
	num_train_examples = num_batches * batch_size

	x_train = x_train[:num_train_examples,:]
	y_train = y_train[:num_train_examples,:]

	# reshape data
	if (lstm_stateful is True):
		x_train = x_train.reshape((num_train_examples, timesteps, lab_dim))
	else:
		x_train = x_train.reshape(num_batches, batch_size, lab_dim)
		y_train = y_train.reshape(num_batches, batch_size, cmp_dim)

	# split train/val
	num_val_examples = nval*batch_size
	x_val = x_train[-num_val_examples:num_train_examples]
	y_val = y_train[-num_val_examples:num_train_examples]
	num_train_examples -= num_val_examples
	x_train = x_train[:num_train_examples]
	y_train = y_train[:num_train_examples]

	return x_train, y_train, x_val, y_val

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--feat_dim', default=configs.feat_dim, help='input feature dimension.')
	parser.add_argument('--num_speakers', default=configs.num_speakers, help='number of speakers.')
	parser.add_argument('--scpfile', default=configs.scpfile, help='scriptfile.')
	parser.add_argument('--batch_size', default=configs.batch_size, help='batch size.')
	parser.add_argument('--timesteps', default=configs.timesteps, help='time steps.')
	parser.add_argument('--lstm_stateful', default=configs.lstm_stateful, help='whether stateful or not should be used.')
	parser.add_argument('--nval', default=configs.nval, help='percentage of samples to be used for validation.')
	parser.add_argument('--datadir', default=configs.datadir, help='directory containing the training data.')
	parser.add_argument('--model_dir', default=configs.model_dir, help='directory to store the models.')
	parser.add_argument('--num_epochs', default=configs.num_epochs, help='number of epochs.')
	args = parser.parse_args()

	# get configs
	feat_dim = args.feat_dim
	num_speakers = args.num_speakers
	scpfile = args.scpfile
	batch_size = args.batch_size
	timesteps = args.timesteps
	lstm_stateful = args.lstm_stateful
	nval = args.nval
	datadir = args.datadir
	model_dir = args.model_dir
	num_epochs = args.num_epochs

        # build model
	model = build_model(feat_dim, num_speakers, batch_size, timesteps, modelid=2, lstm_stateful=lstm_stateful)
	model.summary()

        # output model directory
	try:
		os.stat(model_dir)
	except:
		os.mkdir(model_dir)

	# Load training data
	print('Reading training data...')
#	x_train, y_train, num_train_examples = read_data(scpfile, datadir, feat_dim, num_speakers)
	x_train, y_train, num_train_examples = read_data_v2(scpfile, datadir, feat_dim, num_speakers)
	print(' %d examples have been loaded' % num_train_examples)

	# prepare data for training
	x_train, y_train, x_val, y_val = prepare_data(x_train, y_train, timesteps, batch_size, nval, lstm_stateful=lstm_stateful)
	num_train_examples = x_train.shape[0]
	num_val_examples = x_val.shape[0]
	assert num_train_examples == y_train.shape[0]
	assert num_val_examples == y_val.shape[0]
	print(' %d examples will effectively be used for training' % num_train_examples)
	print(' %d examples will be used for validation' % num_val_examples)

	# show input and output matrices
	print('HERE IS X IN TRAINING FORMAT:\n')
	print(x_train)
	print('AND HERE IS Y IN TRAINING FORMAT:')
	print(y_train)
	print('Dimension of the input matrix : {}'.format(x_train.shape))
	print('Dimension of the output matrix: {}'.format(y_train.shape))

	# define optimization criterion and method
	keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])

	# Train the model, iterating on the data in batches
	print('Start training...\n')
	filepath = model_dir + '/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	# fit the model
	if lstm_stateful is False:
		model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2)
	else:
		for i in range(num_epochs):
			model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=False, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2)
			model.reset_states()

	print('\n\nSUMMARY OF THE PERFORMANCE OF THE MODEL:')
	scores = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=8)
	print("Model Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == "__main__":
	main()
