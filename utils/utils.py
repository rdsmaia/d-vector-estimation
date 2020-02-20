import numpy as np
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed


# read speaker id specific data
def read_data_speakerid(speakerid, params, norm_type='MVN'):

	num_mels = params.mel_dim
	script_file = params.scpfile
	datadir = params.datadir

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
		if int(speaker_ids[i]) == speakerid:
			mel = np.load(os.path.join(datadir,files))
			inputs += mel.tolist()
	assert inputs != []
	inputs = np.asarray(inputs)

	if norm_type == 'MVN':
		mean_var_scaler = preprocessing.StandardScaler()
		inputs = mean_var_scaler.fit_transform(inputs)
	elif norm_type == '-1_to_1':
		inputs = 0.125 * inputs
	elif norm_type == '0_to_1':
		inputts = (inputs + 4.) / 8.
	else:
		raise ValueError(f'Normalization {norm_type} unkown!')

	num_examples = len(inputs)
	return inputs, num_examples



# read mel spectrograms and speaker ids
def read_data(params, norm_type='MVN'):

	num_mels = params.mel_dim
	num_speakers = params.num_speakers
	script_file = params.scpfile
	datadir = params.datadir

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

	inputs = np.asarray(inputs, dtype=np.float16)
	outputs = np_utils.to_categorical(outputs, num_speakers, dtype='float16')

	if norm_type == 'MVN':
		mean_var_scaler = preprocessing.StandardScaler()
		inputs = mean_var_scaler.fit_transform(inputs)
	elif norm_type == '-1_to_1':
		inputs = 0.125 * inputs
	elif norm_type == '0_to_1':
		inputts = (inputs + 4.) / 8.
	else:
		raise ValueError(f'Normalization {norm_type} unkown!')

	assert len(inputs) == len(outputs)

	num_examples = len(inputs)
	return inputs, outputs, num_examples


# prepare data: split test/validation, shuffle or not
def prepare_data_inference(x_train, params):

	timesteps = params.timesteps
	batch_size = params.batch_size
	lstm_stateful = params.lstm_stateful
	mel_dim = params.mel_dim

	# shuffle data
	if lstm_stateful is False:
		x_train = shuffle(x_train, random_state=0)
		
	num_train_examples = x_train.shape[0]
	num_batches = int(np.fix(num_train_examples/batch_size))
	num_train_examples = num_batches * batch_size

	x_train = x_train[:num_train_examples,:] 

	# reshape data
	if (lstm_stateful is True):
		x_train = x_train.reshape((num_train_examples, timesteps, mel_dim))
	else:
		x_train = x_train.reshape(num_batches, batch_size, mel_dim)

	return x_train


# prepare data: split test/validation, shuffle or not
def prepare_data(x_train, y_train, params):

	timesteps = params.timesteps
	batch_size = params.batch_size
	lstm_stateful = params.lstm_stateful
	mel_dim = params.mel_dim
	num_speakers = params.num_speakers
	nval = params.nval

	# shuffle data
	if lstm_stateful is False:
		x_train, y_train = shuffle(x_train, y_train, random_state=0)
		
	num_train_examples = x_train.shape[0]
	num_batches = int(np.fix(num_train_examples/batch_size))
	num_train_examples = num_batches * batch_size

	x_train = x_train[:num_train_examples,:] 
	y_train = y_train[:num_train_examples,:] 

	# reshape data
	if (lstm_stateful is True):
		x_train = x_train.reshape((num_train_examples, timesteps, mel_dim))
	else:
		x_train = x_train.reshape(num_batches, batch_size, mel_dim)
		y_train = y_train.reshape(num_batches, batch_size, num_speakers)

	# split train/val
	num_val_examples = nval*batch_size
	x_val = x_train[-num_val_examples:num_train_examples]
	y_val = y_train[-num_val_examples:num_train_examples]
	num_train_examples -= num_val_examples
	x_train = x_train[:num_train_examples]
	y_train = y_train[:num_train_examples]

	return x_train, y_train, x_val, y_val


# build model.
# NOTE: at the moment the best model is modelid=2
def build_model(params):

	batch_size = params.batch_size
	timesteps = params.timesteps
	mel_dim = params.mel_dim
	num_speakers = params.num_speakers
	modelid = params.modelid

	model = Sequential()

	if modelid == 1:
		# 1DNN512_3DNN1024_1LSTM512_1DNN16_1DNN{num_speakers}
		model.add(TimeDistributed(Dense(512, activation='tanh'), batch_input_shape=(batch_size, timesteps, mel_dim)))
		model.add(TimeDistributed(Dense(1024, activation='tanh')))
		model.add(TimeDistributed(Dense(1024, activation='tanh')))
		model.add(TimeDistributed(Dense(1024, activation='tanh')))
		model.add(LSTM(512, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(Dense(16, activation='tanh'))
		model.add(Dense(num_speakers, activation='softmax'))
	elif modelid == 2:
		# 4DNN1024_1DNN64_1DNN{num_speakers} NOTE: best model so far
		model.add(Dense(1024, activation='tanh', input_shape=(batch_size, mel_dim)))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(1024, activation='tanh'))
		model.add(Dense(64, activation='tanh'))
		model.add(Dense(num_speakers, activation='softmax'))
	elif modelid == 3:
		# 4LSTM512_1DNN64_1DNN512_1DNN{num_speakers}
		model.add(LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', batch_input_shape=(batch_size, timesteps, mel_dim), stateful=True))
		model.add(LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(LSTM(512, activation='tanh', recurrent_activation='sigmoid', stateful=True))
		model.add(Dense(64, activation='tanh'))
		model.add(Dense(512, activation='tanh'))
		model.add(Dense(num_speakers, activation='softmax'))
	else:
		raise ValueError(f'Unknown modelid {modelid}!')

	return(model)

