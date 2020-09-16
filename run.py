from model import *

model = TransformerEncoderBundle(d_model, nhead, dim_feedforward, num_layers, dropout, activation).to(device)
ctc_loss = nn.CTCLoss(blank=vocab_size-1, zero_infinity=False)
#loss = ctc_loss(input, target, input_lengths, target_lengths)

optim = torch.optim.SGD(model.parameters(), lr=lr)
#optim = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optim, 6, gamma=0.95)

def load_batch_data(mini_lab, stage):
	feats_d = str.format(feats_d_format, stage) 

	for line in mini_lab:
		#_id = 'dev-other-feat'+line.split(' ')[0]
		_id = line.split(' ')[0]

		_np = np.load(feats_d + '/' + _id + extension)['arr_0'] #for npz
		#_np = np.load(root + '/' + _id + extension) #for npy
		seq_len = _np.shape[0]
		if seq_len > max_seq_len:
			print(seq_len)
		_np = np.concatenate((_np, np.zeros((max_seq_len-seq_len, d_raw))), axis=0)
		_np = _np.reshape((1, _np.shape[0], _np.shape[1]))

		_lab = ' '.join(line.split(' ')[1:]).upper()
		_lab = re.sub(r'[^A-Z ]', '', _lab)
		tgt_len = len(_lab)

		## change lab to array.
		_meta = np.asarray([_lab, seq_len, tgt_len]).reshape(1,3)
		#_meta = _meta.reshape((1, _meta.shape[0], _meta.shape[1]))

		try:
			inp_data = np.concatenate((inp_data, _np), axis=0)
			loss_data = np.concatenate((loss_data, _meta), axis=0)

		except:
			inp_data = _np
			loss_data = _meta

	return inp_data, loss_data

def run_loss(output, mini_loss):
	#--------- PADDED CTC --------#
	#mini_loss = np.asarray([_lab, seq_len, tgt_len])	
	# transcripts = mini_loss[:, 0]
	# tgts = tokenizer.texts_to_sequences(transcripts) # N x T
	# for tgt in tgts:
	# 	tgt.extend([1]*(max_tgt_length-len(tgt)))
	# tgts = get_tensor(np.array(tgts)).float()

	#---------UNPADDED CTC--------#
	transcripts = [''.join(mini_loss[:, 0])]
	tgts = np.array(tokenizer.texts_to_sequences(transcripts))-1 # N x T
	tgts = get_tensor(tgts.reshape(tgts.shape[1],)).type(torch.long)

	inp_lens = get_tensor(np.array([int(x) for x in mini_loss[:, 1]])).type(torch.long)
	tgt_lens = get_tensor(np.array([int(x) for x in mini_loss[:, 2]])).type(torch.long)

	#print(tgt_lens, inp_lens)
	#print(output.dtype, tgts.dtype, inp_lens.shape, tgt_lens.shape, tgt_lens.sum())

	return ctc_loss(output, tgts, inp_lens, tgt_lens)

# Evalute the model on the validation / dev data
def evaluate(epoch):
	#T, N, d_model, where T is sequence length, N is batch size, d_model is the features
	#x = train.transpose(1,0,2)
	#x_dev = dev.transpose(1,0,2)

	stage = 'dev'
	model.eval()
	total_loss = 0

	print('-' * 89)
	print('.'*35, 'VALIDATION EPOCH {0}'.format(epoch), '.'*35)\

	with torch.no_grad():
		lab_d = str.format(lab_d_format, stage)
		with open(lab_d) as lab_df:
			lab = lab_df.readlines()
			tot_exs = len(lab)

			for batch, i in enumerate(range(0, tot_exs, N)):
				#print("loading batch..", (batch+1))
				end_i = (i+N) if ((i+N) < tot_exs) else tot_exs
				mini_lab = lab[i:end_i]

				mini, mini_loss = load_batch_data(mini_lab, stage)

				mini = get_tensor(mini.transpose(1,0,2)).float().to(device)
				output = model(mini).log_softmax(2).detach()
		
				total_loss += run_loss(output, mini_loss)

	return total_loss / (tot_exs // N)

def train(epoch):
	#T, N, d_model, where T is sequence length, N is batch size, d_model is the features
	#x = train.transpose(1,0,2)
	#x_dev = dev.transpose(1,0,2)

	stage = 'train'
	model.train() #Turn on the train mode.

	total_loss = 0
	start_time = time.time()

	print('-' * 89)
	print('.'*35, 'TRAINING EPOCH {0}'.format(epoch), '.'*35)
	print('-' * 89)

	lab_d = str.format(lab_d_format, stage)

	with open(lab_d) as lab_df:
		lab = lab_df.readlines()
		tot_exs = len(lab)
		#for epoch in range(epochs):
		for batch, i in enumerate(range(0, tot_exs, N)):
			#print("loading batch..", (batch+1))
			end_i = (i+N) if ((i+N) < tot_exs) else tot_exs
			mini_lab = lab[i:end_i]

			mini, mini_loss = load_batch_data(mini_lab, stage)
			#print((batch+1),"...loaded")
			optim.zero_grad()

			mini = get_tensor(mini.transpose(1,0,2)).float().to(device)
			#output = model(mini).log_softmax(2).detach().requires_grad_()
			output = model(mini).detach().requires_grad_()
	
			loss = run_loss(output, mini_loss)

			total_loss += loss.item()
			#print('batch {0}, loss {1}'.format(batch+1, loss.item()))

			loss.backward()

			nn.utils.clip_grad_value_(model.parameters(), 0.5)
			optim.step()

			if batch % log_interval == 0 and batch > 0:
				cur_loss = total_loss / log_interval
				elapsed = time.time() - start_time
				#---------- DEBUGGING --------#
				#print(np.argmax(output.cpu().data.numpy(), axis=2))
				#print(output.shape)
				if batch % print_out_interval == 0:
					model_output = ctc_forward(output, mini_loss[0, 1])
					print("predicted len: ", len(model_output), "\t", "actual len: ", mini_loss[0, 2], "\t" "unique chars: ", len(set(model_output)))
					print(model_output, "\t", mini_loss[0,0])
				#---------- DEBUGGING END--------#
				print('| epoch {:3d} | {:5d}/{:5d} batches | '
					  'lr {:02.2f} | ms/batch {:5.2f} | '
					  'loss {:5.2f} | ppl {:8.2f}'.format(
						epoch, batch, tot_exs // N, scheduler.get_lr()[0],
						elapsed * 1000 / log_interval,
						cur_loss, math.exp(cur_loss)))
				total_loss = 0
				start_time = time.time()

			#print("inp_dim: ",x.shape, "out_dim",output.shape, 
			#	"sum_shape",tgtst.shape,"inp_lens_shape",inp_lens.shape,"tgt_lens_shape",tgt_lens.shape)	

		print()

# Debug method to remove the duplicates, blank and truncate the output
def ctc_forward(output, seq_len):
	output = output.cpu().data.numpy()
	#--------- GREEDY DECODING --------#
	#output shape - T x N x L (sequence x batch x classes)
	# d_output = (np.argmax(output, axis=2)) # T x N
	# d_output = d_output.reshape((d_output.shape[1], d_output.shape[0])) # N x T
	#print(d_output.shape)
	#for index in range(d_output.shape[0]):	
	# model_output = ''.join(('-' if num==0 else tokenizer.index_word[num]) for num in d_output)
	# model_output = model_output[:int(seq_len)]
	# model_output = (''.join(i for i, _ in itertools.groupby(model_output)))
	# model_output = model_output.replace('-', '')

	#------ BEAM DECODER - 1st Output ------#
	#output shape - T x N x L (sequence x batch x classes)
	d_output = output.transpose((1,0,2))[0] # T x L
	d_output = d_output.reshape((d_output.shape[0], 1, d_output.shape[1])) # T x 1 x L
	decoded, log = tf.nn.ctc_beam_search_decoder(d_output, np.array([seq_len]), beam_width=100, top_paths=1)
	#indices = decoded[0].indices.cpu().numpy()
	#print(indices.reshape((indices.shape[1], indices.shape[0])))
	d_output = decoded[0].values.cpu().numpy()
	#print(d_output)
	model_output = ''.join(('-' if num==vocab_size-1 else tokenizer.index_word[num+1]) for num in d_output)
	return model_output

def get_tensor(np):
	return Variable(torch.from_numpy(np))

if __name__=="__main__":
	best_val_loss = float("inf")
	best_model = None

	for epoch in range(epochs):
		epoch_start_time = time.time()
		train(epoch)
		val_loss = evaluate(epoch)
		#val_loss=0
		print('-' * 89)
		print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
		print('-' * 89)
		
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model = model
			torch.save(best_model.state_dict(), best_model_path)

		scheduler.step()