import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.autograd import Variable
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import time
import itertools

#from importlib import reload

tokens = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'

############ Tokenizer ###########

dataset = [tokens]
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(dataset)

############ Data IO ############ 

vocab_size = len(tokens) + 1
d_softmax = 2

#feats directory
#feats_d_format = 'dataset/LibriSpeech/{0}_feat' #Local
feats_d_format = 'data/feats/{0}' #GPU

#labels files
#lab_d_format = 'dataset/LibriSpeech/{0}_lab_dv2.txt' #Local
lab_d_format = 'data/lab/{0}_lab.txt' #GPU

max_seq_len = 1450 #180 with feature vector size of 3200
max_tgt_length = 620

extension = '.npz'

########### Model Params ###########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#T, N, d_model, where T is sequence length, N is batch size, d_model is the features
N = 8
d_model = 128
d_raw = 400

nhead = 8
dim_feedforward = 2048
num_layers = 8 
dropout=0.5
lr = 1
activation='relu'

epochs = 100

best_model_path = 'model'

######## Console Printing #########

log_interval = 25
print_out_interval = 1000
