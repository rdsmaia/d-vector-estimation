# D-vector training/estimation

The method consists in the one described at

[1] R. Doddipatla et al. "Speaker Adaptation in DNN-Based Speech Synthesis Using d-Vectors", Proc. of INTERSPEECH, 2017.

which consists in training a NN under a speaker identification task with a bottleneck. The average or PCA of the averaged bottleneck serves as d-vectors.

# Quick start: training

- place your mel spectrogram files in a specific location, e.g. "data"
- create a scriptfile list with two columns, where in the first colums you have the mel spectrograms filename and in the second column the speaker id.
- adjust configs in train.py
- run:
$ python train.py

# Quick start: estimation of d-vectors

- adjust configs in get_speakerid.py
- run:
$ python get_speakerid.py --speakerid='(speaker_id)' --out_dvector_file='(dvector_filename.npy)' --model_path='(path_to_saved_model.hdf5)'

# Some notes

- at the moment the best model id is 3, which uses a stack of LSTM layers. However, training time is a drawback. Model id 2 is good enough and fairly fast to train.
- There are two ways of calculating d-vectors: mean and PCA. Mean takes the frame-by-frame mean of the bottleneck layer while PCA takes the PCA. So far for the experiments done "mean" achieves the best result.

# To do:

- Training on batches. At the moment all the data is loaded at the memory. This can be an issue for very large databases/small meory machines. Maybe a proper data generator should be used.
- Moving away from this approach, the autoencoding method could be implemented, as described in the paper below:
[2] Ye Jia et al. "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis," 32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montreal, Canada.

