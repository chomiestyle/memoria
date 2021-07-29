
"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

"""
Model Training 
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""

import logging
import coloredlogs
import matplotlib.pyplot as plt
from docopt import docopt
import matplotlib.gridspec as gridspec
from rainbow.trading_bot.agent import Agent
from rainbow.trading_bot.methods import train_model, evaluate_model
from rainbow.trading_bot.utils import (get_stock_data_1d,get_stock_data_1m,format_currency,format_position,show_train_result,switch_k_backend_device)
from Forecasting.get_syntetic_data import *

from Forecasting.LSTM.LSTM import Predict
from Forecasting.TimeGAN.TGAN_future import *



def train_bot(train_stock, val_stock, window_size, batch_size, ep_count,per_value,
				 strategy="t-dqn", model_name="model_debug", pretrained=False,
				 debug=False):
		""" Trains the stock trading bot using Deep Q-Learning.
		Please see https://arxiv.org/abs/1312.5602 for more details.

		Args: [python train.py --help]
		"""
		#Se inicializa el agente
		agent = Agent(2 * window_size - 1, strategy=strategy, pretrained=pretrained, model_name=model_name, manual=True)
		#Se crean los arreglos de variables
		final_rewards = []
		train_roi = []
		valid_roi = []
		train_loss = []
		rewards = []
		name_stocks = ['AAPL', 'CSCO', 'CVX', 'IBM', 'INTC', 'MSFT', 'GS']
		total_episode=1
		total_ep_count=ep_count*len(name_stocks)
		for stock in name_stocks:
			print('stock : {}'.format(stock))
			n_seq = 8
			seq_len = window_size
			tgan_model_path = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/TimeGAN/TimeGAN_random_input_{}/experiment_00/synthetic_data'.format(stock)
			lstm_model_path = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/lstm_{}.h5'.format(stock)
			gan_model = TGAN(train_step=69, save_directory='TimeGAN_random_input_{}'.format(stock))
			tgan=gan_model.training_loop(stock_name=stock)

			#Se entrena la red LSTM
			predict = Predict()
			x_train, y_train, scaler, time = predict.prepare_data(SYMBOL=stock)
			lstm_model = predict.train(x_train, y_train, is_first=True)
			lstm_model.save(lstm_model_path,include_optimizer=True,save_format='h5')
			#Prepare training data with future information
			gen_train, real_train, prev_train, pred_train, t_data_train = get_syntetic_data(symbol=stock, tipo='train', n_seq=n_seq, seq_len=seq_len,
															  gen_model=tgan, lstm_model=lstm_model)
			gen_train_data, gen_real_value = join_data_g(generated_data=gen_train, prev_data=prev_train, real_data=real_train)
			pred_train_data, pred_real_data = join_data_g(generated_data=pred_train, prev_data=prev_train, real_data=real_train)

			#Prepare training data with future information
			gen_val, real_val, prev_val, pred_val, t_data_val = get_syntetic_data(symbol=stock, tipo='test', n_seq=n_seq, seq_len=seq_len,
															  gen_model=tgan, lstm_model=lstm_model)
			gen_val_data, gen_real_value = join_data_g(generated_data=gen_val, prev_data=prev_val, real_data=real_val)
			pred_val_data, pred_real_data = join_data_g(generated_data=pred_val, prev_data=prev_val, real_data=real_val)

			# if per_value=='minute':
			# 	train_data = get_stock_data_1m(train_stock)
			# 	val_data = get_stock_data_1m(val_stock)
			# elif per_value=='day':
			# 	train_data = get_stock_data_1d(train_stock)
			# 	val_data = get_stock_data_1d(val_stock)
			# else:
			# 	print('Tienes que avisarme si el entrenamiento es por dia (per_value=day) o por minuto (per_value=minute)')
			# 	return

			#initial_offset = val_data[1] - val_data[0]


			for episode in range(1, ep_count + 1):

					train_result,rewards = train_model(agent, total_episode, pred_train_data, ep_count=total_ep_count,
																		 batch_size=batch_size)
					final_rewards.extend(rewards)
					train_roi.append(train_result[2])
					train_loss.append(train_result[3])
					val_result, _ = evaluate_model(agent, pred_val_data, window_size, debug)
					print('profits generados LSTM en {} : {}'.format(stock,val_result))
					valid_roi.append(val_result)

					train_result,rewards = train_model(agent, total_episode, gen_train_data, ep_count=total_ep_count,
																		 batch_size=batch_size)
					final_rewards.extend(rewards)
					train_roi.append(train_result[2])
					train_loss.append(train_result[3])
					val_result, _ = evaluate_model(agent, gen_val_data, window_size, debug)
					print('profits generados con TGAN en {} : {}'.format(stock,val_result))
					valid_roi.append(val_result)
					total_episode += 1
					#show_train_result(train_result, val_result, initial_offset)

		gs = gridspec.GridSpec(2, 2)
		fig = plt.figure(figsize =(20,9))
		
		# To be shifted to Axis 1
		ax1 = fig.add_subplot(gs[0, 0])
		ax1.plot(range(len(train_loss)), train_loss, color='purple', label='loss')
		ax1.legend(loc=0, ncol=2, prop={'size':20}, fancybox=True, borderaxespad=0.)
		ax1.set_xlabel('Epochs', size=20)
		ax1.set_ylabel('Train Loss', size=20)
		ax1.set_title('Loss w.r.t. Epochs', size=20)

		# To be shifted to Axis 2
		ax2 = fig.add_subplot(gs[0, 1])
		ax2.plot(range(len(train_roi)),train_roi, color = 'crimson', label='train')
		ax2.plot(range(len(valid_roi)), valid_roi, color='olive', label='val')
		ax2.legend(loc=0, ncol=2, prop={'size':20}, fancybox=True, borderaxespad=0.)
		ax2.set_ylabel('Return of Investment($)', size=20)
		ax2.set_xlabel('Epochs', size=20)
		ax2.set_title('Train and Valid ROI w.r.t. Epochs', size=20)

		# To be shifted to Axis 3
		ax3 = fig.add_subplot(gs[1, :])
		ax3.plot(range(len(final_rewards)),final_rewards, color='red', label = 'Reward of Rainbow DQN')
		ax3.set_xlabel('Episodes', size=20)
		ax3.set_ylabel('Rewards', size=20)
		ax3.set_title('Reward w.r.t. episodes', size=20)
		ax3.legend(loc=0, ncol=2, prop={'size':20}, fancybox=True, borderaxespad=0.)

		plt.show()






train_stock='C:/Users/eduar/PycharmProjects/Comparacion_memoria/rainbow/data/day/train/AAPL_train.csv'
val_stock='C:/Users/eduar/PycharmProjects/Comparacion_memoria/rainbow/data/day/train/AAPL_test.csv'
strategy='double-dqn'
window_size = 5
batch_size = 10
ep_count = 10
debug=False
coloredlogs.install(level="DEBUG")
per_value='minute'
switch_k_backend_device()
model_name='double-dqn_1D_with_prediction'
try:
	train_bot(train_stock=train_stock, val_stock=val_stock, window_size=window_size,
			  batch_size=batch_size,ep_count=ep_count,per_value=per_value, strategy=strategy,model_name=model_name,pretrained=False, debug=debug)
except KeyboardInterrupt:
	print("Aborted!")

