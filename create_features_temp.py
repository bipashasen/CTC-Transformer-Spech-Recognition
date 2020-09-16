import soundfile as sf 
import os
import numpy as np
from numpy import asarray 
from numpy import save
from numpy import savez
import sys

folder_path = sys.argv[1]
features_folder_path = sys.argv[2]
featureAudioMapping = sys.argv[3]
feature_size = 3200

def processFeatures():
	featureAudioMap = list()
	maxFrameLength = -1
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			if file.endswith('.flac'):
				# read the audio file and store the features 
				# in features_folder_path with the same name as audio_file 
				audioFilePath = root + '/' + file
				data, samplerate = sf.read(audioFilePath)
				data = np.append(data, [0]*(feature_size-data.shape[0]%feature_size))
				data = data.reshape(int(data.shape[0]/feature_size), feature_size)
				if data.shape[0] > maxFrameLength:
					maxFrameLength = data.shape[0]
					maxAudioFile = audioFilePath
				featureFilePath = features_folder_path + file.split('.')[0] + '.npz'
				savez(featureFilePath, data)
				#featureFilePath = features_folder_path + file.split('.')[0] + '.npy'
				#save(featureFilePath, data)
				# add the mapping to a file
				featureAudioMap.append(audioFilePath + ' ' + featureFilePath)

	print('Max sequence length : ' + str(maxFrameLength))
	print('Max audio file : ' + maxAudioFile)
	# save the mapping into a file 
	with open(featureAudioMapping, 'w') as f:
		for line in featureAudioMap:
			f.write(line + '\n')


if __name__ == "__main__":
	processFeatures()
