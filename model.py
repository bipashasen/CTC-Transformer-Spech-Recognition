from constants import *

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.in_channel = 1
		channel_lay1 = 64
		channel_lay2 = 128
		channel_lay3 = 256

		kernel_lay1 = 40
		kernel_lay2 = 20
		kernel_lay3 = 10

		self.maxp_lay1 = 3
		self.maxp_lay2 = 2
		self.maxp_lay3 = 2

		self.batch_lay1 = nn.BatchNorm1d(channel_lay1, momentum=0.05)
		self.batch_lay2 = nn.BatchNorm1d(channel_lay2, momentum=0.05)
		self.batch_lay3 = nn.BatchNorm1d(channel_lay3, momentum=0.05)

		self.ReLU = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

		#self.cnn_lay1 = nn.Conv1d(in_channel, channel_lay1, (d_raw-d_model+1))
		self.cnn_lay1 = nn.DataParallel(nn.Conv1d(self.in_channel, channel_lay1, kernel_lay1))
		self.cnn_lay2 = nn.DataParallel(nn.Conv1d(channel_lay1, channel_lay2, kernel_lay2))
		self.cnn_lay3 = nn.DataParallel(nn.Conv1d(channel_lay2, channel_lay3, kernel_lay3))

		self.linear = nn.Linear(channel_lay3*20, d_model) #20 is the output feature dimension calculated manully on the above parameters.

	def forward(self, x_in):
		#T x N x d_model -> (T*N), inp_channels, d_model
		x = x_in.reshape((x_in.shape[0]*x_in.shape[1], self.in_channel, x_in.shape[2]))

		#self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))
		x_lay1 = self.dropout(self.ReLU(self.batch_lay1(F.max_pool1d(self.cnn_lay1(x), self.maxp_lay1))))
		x_lay2 = self.dropout(self.ReLU(self.batch_lay2(F.max_pool1d(self.cnn_lay2(x_lay1), self.maxp_lay2))))
		x_lay3 = self.dropout(self.ReLU(self.batch_lay3(F.max_pool1d(self.cnn_lay3(x_lay2), self.maxp_lay3))))

		x = x_lay3.view(x_lay3.size()[0], -1)
		x = self.linear(x) #N' x d is the output
		#print(x_lay1.shape, " ", x_lay2.shape, " ", x_lay3.shape, " ", x.shape)
		
		#(T*N), out_channels, d_model -> T x N x d_model
		x = x.reshape((x_in.shape[0], x_in.shape[1], x.shape[1]))
		#print(x.shape)

		return x


class TransformerEncoderBundle(nn.Module):
	def __init__(self, d_model, nhead=4, dim_feedforward=1024, num_layers=6, dropout=0.5, activation='relu'):
		super(TransformerEncoderBundle, self).__init__()
		#torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
		#d_model – the number of expected features in the input (required).
		#nhead – the number of heads in the multiheadattention models (required).
		#dim_feedforward – the dimension of the feedforward network model (default=2048).
		#dropout – the dropout value (default=0.1). 
		#activation – the activation function of intermediate layer, relu or gelu (default=relu).
		# embed_dim must be divisible by num_heads
		self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) 

		self.encoder_norm = nn.LayerNorm(d_model)

		#torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
		#encoder_layer – an instance of the TransformerEncoderLayer() class (required).
		#num_layers – the number of sub-encoder-layers in the encoder (required).
		#norm – the layer normalization component (optional).
		self.encoder = nn.DataParallel(nn.TransformerEncoder(self.encoder_layer, num_layers, self.encoder_norm))

		self.encode_raw = nn.Linear(d_raw, d_model)
		self.encode_cnn = CNN()

		self.linear = nn.Linear(d_model, vocab_size)
		self.softmax = nn.LogSoftmax(d_softmax)
		self.ReLU = torch.nn.ReLU()		

		self.pos_encoder = PositionalEncoding(d_model, dropout)

		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.linear.bias.data.zero_()
		self.linear.weight.data.uniform_(-initrange, initrange)

	def forward(self, x_in):
		#x = self.ReLU(self.encode_raw(x))
		x = self.encode_cnn(x_in)
		#x = self.pos_encoder(x)
		#print("x.shape: ", x.shape)
		output = self.encoder(x)
		#print("out.shape", output.shape)
		output = self.linear(output)
		#print("out.shape", output.shape)
		output = self.softmax(output)
		return output

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		#print(x.shape, " ", self.pe[:x.size(0), :].shape)
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)