import sys
sys.path.insert(0,'rainbow/')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from rainbow.eval import main
from Forecasting.get_syntetic_data import *


def preprocess_data(data, scores):
	print(data.iloc[:,[0,1,2,3,4,5]])
	##Agregar de mejor manera la informacion sentimiento
	minmax_for = MinMaxScaler().fit(data.iloc[:,[0,1,2,3,4,5]].astype('float32')) # Close index, Volume and Sentiment
	df_log = minmax_for.transform(data.iloc[:,[0,1,2,3,4,5]].astype('float32')) # Close index, Volume and Sentiment
	df_log = pd.DataFrame(df_log)
	return df_log,minmax_for

def database_get_data(stock,combined=True):
	print('Currently Pulling',stock)
	CRYPTO_STOCKS = ['BTC-USD','ETH-USD','BCH-USD']
	DOWJONES_STOCKS = ['AAPL']
	if not combined:
		if stock in CRYPTO_STOCKS:
			dir_path= 'C:/Users/56979/PycharmProjects/Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN/Database/Crypto'
			split_name=stock.split('-')
			file_name=split_name[0]+split_name[1]
		elif stock in DOWJONES_STOCKS:
			dir_path= 'C:/Users/56979/PycharmProjects/Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN/Database/DOWJONES'
			file_name=stock
		else:
			print('No existe ese Stock en la base de datos')
			return pd.DataFrame()
		features = ['Open','High', 'Low','Close', 'Volume']
		index_column=1

	else:
		if stock in CRYPTO_STOCKS:
			dir_path= 'C:/Users/56979/PycharmProjects/Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN/Database/Crypto'
			features = ['Open', 'High', 'Low', 'Close', 'Volume','sentiment']
		elif stock in DOWJONES_STOCKS:

			dir_path='C:/Users/eduar/PycharmProjects/Real_Time_prediction/Sentiment/news_prices'
			features = ['Open', 'High', 'Low', 'Close', 'Volume', 'title', 'summary', 'fin_title', 'fin_summary','fin_text']

			file_name ='{}_news_prices'.format(stock)
			index_column = 0

	#Extract_data
	file_path = dir_path + '/{}.csv'.format(file_name)
	data=pd.read_csv(file_path,index_col=index_column,parse_dates=True)
	data=data[features].tail(100)
	data=data.rename(columns={"Open": "1. open","High": "2. high", "Low": "3. low", "Close": "4. close","Volume":"5. volume"})
	data['date'] = data.index
	data['date'] = data['date'].map(mdates.date2num)
	return data




def get_list(data):
	# importing the module
	import itertools
	# flattening the list and storing the result
	flat_list = itertools.chain(*data)
	# converting iterable to list and printing
	total_list=list(flat_list)
	return total_list

def join_data_p(generated_data, prev_data, real_data):
	final_generated = []
	final_real = []
	for i in range(len(generated_data)):
		# print(' sequence:{}'.format(i))
		generated = generated_data[i]
		prev = prev_data[i, :, 3]
		print('previous')
		print(prev)
		generated = np.append(prev, generated, axis=0)
		if i <= len(real_data) - 1:
			real = real_data[i,:,3]
			real = np.append(prev, real, axis=0)
			final_real.append(real)
			final_generated.append(generated)
	return final_generated, final_real




def trading_experiment(stock):
	manual_run = False
	n_seq = 8
	seq_len = 5
	model_path_1 = 'C:/Users/eduar/PycharmProjects/TimeGAN/tensorflow2_implementation/TimeGAN_future2_noise_past1200/experiment_00/synthetic_data'
	val_stock = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/rainbow/data/day/train/AAPL_test.csv'
	path_model = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/lstm.h5'
	gen, real, prev, pred,t = get_syntetic_data(n_seq=n_seq, seq_len=seq_len, model_path=model_path_1)

	generated_data, real_data = join_data_g(generated_data=pred, prev_data=prev, real_data=real)
	real_profit_total=0
	gen_profit_total=0
	for i in range(len(generated_data)):
		generated = generated_data[i]
		real = real_data[i]
		arr = np.array(t[i])
		didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
		gen_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
		real_df = pd.DataFrame(data=real, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
		#print(real_df)
		real_df=real_df.head(5)
		#print(real_df)
		g=list(gen_df['Close'])
		r=list(real_df['Close'])
		#print(r)
		gen_profit, gen_history ,agent_1 = main(g, 5, "AAPL_1D_tdqn_1", True, manual_run, per_value='day',is_prediction=True)
		print(gen_history)
		#print(agent.inventory)
		real_profit, real_history ,agent_2= main(r, 5, "AAPL_1D_tdqn_1", True, manual_run, per_value='day',is_prediction=False)
		print(real_history)
		#print(agent.inventory)
		gen_profit_total+=gen_profit
		real_profit_total+=real_profit
	print('ganancias con el agente y valores reales')
	print(real_profit_total)
	print('ganancias con el agente y valores generados')
	print(gen_profit_total)

	#print(history)

#trading(stock='AAPL')