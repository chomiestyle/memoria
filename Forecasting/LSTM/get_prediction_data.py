import pandas as pd
import mplfinance as mpf
from tensorflow.keras.models import load_model
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from Forecasting.TimeGAN.TGAN_future import batch_noise_structure
from yahoo_fin import *
from yahoo_fin.stock_info import *
import arrow
import numpy as np



def set_datetime(today,type):
    if type=='stocktwits':
        utc_datetime = arrow.get(today)
        utc_datetime=utc_datetime.to('local').to('utc')
    elif type=='price':
        utc_datetime = arrow.get(today).shift(hours=-5)
    date_time_obj = utc_datetime.to('America/Santiago')
    today=date_time_obj.datetime
    tt = today.timetuple()
    date_val = datetime.datetime(year=tt.tm_year, month=tt.tm_mon, day=tt.tm_mday, hour=tt.tm_hour, minute=tt.tm_min)
    return date_val

def prepare_data( seq_len=20,n_seq=5,plot_data=False, batch_size=30):
    # PREPARE DATA
    def get_full_data(SYMBOL, DIR_PATH='C:/Users/eduar/PycharmProjects/TimeGAN/Data/finbert_sentiment', n_values=None):
        search_path = DIR_PATH + '/results_{}_new.csv'.format(SYMBOL)
        index_column = 0
        dataframe = pd.read_csv(search_path, index_col=index_column, parse_dates=True)
        if n_values != None:
            dataframe = dataframe.tail(n_values)
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)
        dataframe['datetime'] = dataframe.index
        dataframe['datetime'] = dataframe['datetime'].apply(lambda x: set_datetime(today=x, type='price'))
        time_index = dataframe['datetime'].values
        features = ['open', 'high', 'low', 'close', 'volume', 'adjclose', 'avg_sentiment_mean',
                    'avg_sentiment_variance']
        data = dataframe[features]
        return data, time_index

    df, time_index = get_full_data(SYMBOL='AAPL')
    if plot_data:
        # candle_data=candle_format(data=self.df)
        mpf.plot(df, type='candle')

    ##Normalizan los datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df).astype(np.float32)
    print(scaled_data)
    time_data = []
    ##Dividen los datos en secuencias
    seq_len = seq_len
    data = []
    for i in range(len(scaled_data) - 2 * seq_len):
        data.append(scaled_data[i:i + 2 * seq_len])
        time_data.append(time_index[i:i + 2 * seq_len])

    data_X = []
    data_Z = []
    for seq in data:
        data_Z.append(seq[:seq_len])
        data_X.append(seq[seq_len:])

    ##Numero de secuencias
    n_windows = len(data)
    print('n_windows:{}'.format(n_windows))

    prev_series = (tf.data.Dataset.from_tensor_slices(data_Z).batch(batch_size))
    prev_series_iter = iter(prev_series.repeat())


    ##Generador de data

    x_series = (tf.data.Dataset.from_tensor_slices(data_X).batch(batch_size))
    x_series_iter = iter(x_series.repeat())

    return x_series_iter,prev_series_iter,scaler,time_data


def get_predicted_data(n_seq,seq_len,model_path):

    x_iter,prev_iter,scaler,t_data=prepare_data(seq_len=seq_len,n_seq=n_seq)
    Z_ = next(prev_iter)
    R_=next(x_iter)
    # #Obtenemos los valores del LSTM

    lstm_model = load_model(model_path)
    predicted_data = lstm_model.predict(Z_)
    print(predicted_data)

    prev_data=np.array(Z_)
    real_data=np.array(R_)

    #pred_data=scaler.inverse_transform(predicted_data)
    #print(pred_data)
    #gen_data = (scaler.inverse_transform(gen_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
    real_data = (scaler.inverse_transform(real_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
    prev_data = (scaler.inverse_transform(prev_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
    return real_data,prev_data,t_data


path_model = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/lstm.h5'

r,p,t=get_predicted_data(n_seq=8,seq_len=5,model_path=path_model)