import soundfile as sf 
import os
import numpy as np
from numpy import asarray 
from numpy import save
from numpy import savez_compressed

import sys

folder_path = sys.argv[1]
features_folder_path = sys.argv[2]
featureAudioMapping = sys.argv[3]

feature_size = 400

def processFeatures():
	filesProcessed = 0
	if not os.path.exists(features_folder_path):
		os.mkdir(features_folder_path)
	maxSequenceLength = -1
	featureAudioMap = list()
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			if file.endswith('.flac'):
				# read the audio file and store the features 
				# in features_folder_path with the same name as audio_file 
				audioFilePath = root + '/' + file
				data, samplerate = sf.read(audioFilePath)
				data = np.append(data, [0]*(feature_size-data.shape[0]%feature_size))
				data = data.reshape(int(data.shape[0]/feature_size), feature_size)
				if data.shape[0] > maxSequenceLength:
					maxSequenceLength = data.shape[0]
					maxAudioFile = audioFilePath
				#featureFilePath = features_folder_path + file.split('.')[0] + '.npy'
				#save(featureFilePath, data)
				featureFilePath = features_folder_path + file.split('.')[0] + '.npz'
				savez_compressed(featureFilePath, data)
				# add the mapping to a file
				featureAudioMap.append(audioFilePath + ' ' + featureFilePath)
				filesProcessed += 1
			if (filesProcessed % 1000) ==0:
				print('Files processed : ' + str(filesProcessed))

	print('The maximum sequence length is : ' + str(maxSequenceLength))
	print('Audio file : ', maxAudioFile)
	# save the mapping into a file 
	with open(featureAudioMapping, 'w') as f:
		for line in featureAudioMap:
			f.write(line)


if __name__ == "__main__":
	processFeatures()
