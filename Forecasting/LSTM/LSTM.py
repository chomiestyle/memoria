import numpy

from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input,LSTM,Dropout
import mplfinance as mpf
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from yahoo_fin.stock_info import *
import arrow
import numpy as np

import Forecasting.LSTM.LSTM


def set_datetime(today, type):
    if type == 'stocktwits':
        utc_datetime = arrow.get(today)
        utc_datetime = utc_datetime.to('local').to('utc')
    elif type == 'price':
        utc_datetime = arrow.get(today).shift(hours=-5)
    date_time_obj = utc_datetime.to('America/Santiago')
    today = date_time_obj.datetime
    tt = today.timetuple()
    date_val = datetime.datetime(year=tt.tm_year, month=tt.tm_mon, day=tt.tm_mday, hour=tt.tm_hour,
                                 minute=tt.tm_min)
    return date_val

def make_GRU(n_layers, hidden_units, output_units, name):
    return Sequential([GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRU_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)

def make_LSTM(n_layers, hidden_units, output_units,dropout,dropout_r):
    return Sequential([LSTM(units=hidden_units,
                            return_sequences=True,activation='tanh',recurrent_activation='sigmoid',
                            use_bias=True, kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal',dropout=dropout, recurrent_dropout=dropout_r) for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid')])



# def make_LSTM(n_layers, hidden_units, output_units,dropout, name):
#     return Sequential([LSTM(units=hidden_units,
#                             return_sequences=True,
#                             name=f'LSTM_{i + 1}') for i in range(n_layers)] +
#                       [Dropout(rate=dropout,name='Dropout')]+
#                       [Dense(units=output_units,
#                              activation='sigmoid',
#                              name='OUT')], name=name)

def calculate_accuracy(y_true, y_pred):
    real = y_true + 1
    predict = y_pred + 1
    cuadrado=tf.square((real - predict) / real)
    axis = list(range(len(cuadrado.get_shape()) - 1))
    mean, var = tf.nn.moments(x=cuadrado, axes=axis)
    raiz=tf.sqrt(mean)
    percentage = 1 - raiz
    return percentage * 100

num_layers = 1
size_layer = 128
timestamp = 4
epoch = 69
dropout_rate = 0.8
test_size = 10
learning_rate = 0.01

class Predict:
    def __init__(self):
        self.seq_len = 5
        #self.in_neurons=8
        #self.out_neurons=8
        # self.hidden_neurons = 452
        self.batch_size = 5
        self.epochs = 5
        # #self.percentage = 0.8
        # self.lr=0.01
        # self.n_layers=1
        # self.dropout=0.778554
        self.trainning=True
        #self.path_model='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/lstm.h5'


    def prepare_data(self,SYMBOL,OUTPUT_FEATURES,DIR_DATA,is_separated=False, plot_data=False):
            # PREPARE DATA
        def get_data(n_values=None):
                search_path = DIR_DATA + '/{}.csv'.format(SYMBOL)
                index_column = 0
                dataframe = pd.read_csv(search_path, index_col=index_column, parse_dates=True)
                if n_values != None:
                    dataframe = dataframe.tail(n_values)
                dataframe.index = pd.to_datetime(dataframe.index, utc=True)
                dataframe['datetime'] = dataframe.index
                dataframe['datetime'] = dataframe['datetime'].apply(lambda x: set_datetime(today=x, type='price'))
                time_index = dataframe['datetime'].values
                data = dataframe.drop(['datetime'], axis=1)
                # features = ['open','high','low','close','volume','adjclose','avg_sentiment_mean','avg_sentiment_variance']
                # data = dataframe[features]
                return data, time_index

        df,time_index=get_data()
        if plot_data:
            # candle_data=candle_format(data=self.df)
            mpf.plot(df, type='candle')
        if len(OUTPUT_FEATURES)!=len(df.columns):
            self.df=df[OUTPUT_FEATURES]
        else:
            self.df=df

        ##Dividen los datos en secuencias
        #self.n_seq=len(self.df.columns)
        ##Normalizan los datos
        self.scaler = MinMaxScaler()
        scaled_real = self.scaler.fit_transform(self.df).astype(np.float32)
        df_gen =df.copy()
        self.in_neurons=len(df_gen.columns)
        self.out_neurons=len(self.df.columns)
        self.scaler_z = MinMaxScaler()
        scaled_Z_data = self.scaler_z.fit_transform(df_gen).astype(np.float32)

        ##Dividen los datos en secuencias
        time_data_real=[]
        data_real = []
        data_z = []
        if is_separated:
            n_win = int(len(self.df) / (2 * self.seq_len))
            pivot = 0
            for i in range(n_win):
                data_z.append(scaled_Z_data[pivot:pivot + 2 * self.seq_len])
                data_real.append(scaled_real[pivot:pivot + 2 * self.seq_len])
                time_data_real.append(time_index[pivot:pivot + 2 * self.seq_len])
                pivot += 2 * self.seq_len
        else:
            for i in range(len(self.df) - 2 * self.seq_len):
                data_z.append(scaled_Z_data[i:i + 2 * self.seq_len])
                data_real.append(scaled_real[i:i + 2 * self.seq_len])
                time_data_real.append(time_index[i:i + 2 * self.seq_len])
        data_X = []
        data_Z= []
        data_P = []
        print('windows: {}'.format(len(data_real)))
        for i in range(len(data_real) - 1):
            seq_z = data_z[i]
            seq_x = data_real[i]
            if len(seq_z[:self.seq_len]) == self.seq_len and len(seq_x[self.seq_len:]) == self.seq_len:
                data_Z.append(seq_z[:self.seq_len])
                data_P.append(seq_x[:self.seq_len])
                data_X.append(seq_x[self.seq_len:])
        data_P=np.array(data_P)
        data_X = np.array(data_X)
        data_Z= np.array(data_Z)
        self.time = time_data_real
        return data_P,data_X


    # do learning
    def train(self, x_train, y_train,is_first=False):
        # custom_params = {'num_layers': 5, 'dropout': 0.01, 'num_neurons': 1024,
        #                   'learning_rate': 0.1, 'dropout_r': 0.771132}
        custom_params = {'num_layers': 3, 'dropout': 0.1, 'num_neurons': 1024,
                          'learning_rate': 0.46143, 'dropout_r': 0.1}
        if is_first:
            num_neurons = custom_params['num_neurons']
            dropout = custom_params['dropout']
            dropout_r = custom_params['dropout_r']
            num_layers = custom_params['num_layers']
            epsilon = 1e-9
            # learning_rate=0.001
            Model = make_LSTM(n_layers=num_layers, hidden_units=num_neurons, output_units=self.out_neurons,
                                  dropout=dropout, dropout_r=dropout_r)
            lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, custom_params['learning_rate'], 1e-5, 2)
            optimizeAdam = Adam(learning_rate=lr_fn, beta_1=0.9, beta_2=0.999, amsgrad=True, decay=1e-6,
                                epsilon=epsilon)
            Model.compile(loss=MeanSquaredError(), optimizer=optimizeAdam, metrics=[calculate_accuracy])
            # Model = make_LSTM(n_layers=self.n_layers, hidden_units=self.hidden_neurons, output_units=self.out_neurons,dropout=self.dropout,
            #                   name='prediction')
            # optimizeAdam = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=False,
            #                     decay=1e-6)
            # Model.compile(loss=MeanSquaredError(), optimizer=optimizeAdam,metrics=[calculate_accuracy])

        # else:
        #     Model = load_model(predict.path_model,custom_objects={'calculate_accuracy': calculate_accuracy})


        Model.fit(x_train, y_train,  epochs=self.epochs,shuffle=True)
        Model.summary()
        return Model


if __name__ == "__main__":

    predict = Predict()
    #name_stocks=['AAPL']
    # OUTPUT_FEATURES=['close']
    # #OUTPUT_FEATURES = ['open', 'high', 'low', 'close']
    # DATA_PATH = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Data/combined/'
    # TYPE_DATA = 'Most_Important_Features/'
    # #TYPE_DATA = 'Full_data/'
    # #TYPE_DATA = 'data_assets_TI/'
    # #TYPE_DATA = 'OHLC_NEWS/'
    #
    #
    # if len(OUTPUT_FEATURES)>1:
    #     TYPE_OUTPUT = 'OHLC'
    # else:
    #     TYPE_OUTPUT='CLOSE'
    # # do learning, prediction, show to each stock
    # is_first=True
    # for istock in name_stocks:
    #     x_train, y_train = predict.prepare_data(SYMBOL=istock,OUTPUT_FEATURES=OUTPUT_FEATURES,DIR_DATA=DIR_DATA)
    #     # #x_test, y_test = predict.load_data_test(data.iloc[split_pos:], predict.length_of_sequences)
    #     model = predict.train(x_train, y_train,is_first=is_first)
    #     # Guardar el Modelo
    #     model.save(SAVE_DIR+TYPE_DATA+istock+'/'+'lstm_{}.h5'.format(TYPE_OUTPUT),include_optimizer=True,save_format='h5')
    #     is_first=False

    DATA_PATH = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Data/combined/'
    #TYPE_DATA_ARRAY = ['Full_data/', 'Most_Important_Features/', 'data_assets_TI/', 'OHLC_NEWS/']
    #TYPE_DATA_ARRAY = ['OHLC_NEWS/']
    TYPE_DATA_ARRAY = ['Complete/']
    #TYPE_DATA_ARRAY = ['data_assets_TI/']
    TYPE_INPUT = 'PREV_WINDOW'
    # SYMBOL='AAPL'
    OUTPUT_FEATURES = ['close']
    SAVE_DIR = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/TRAIN/'
    OUTPUT_TYPE = 'CLOSE'
    is_first=True
    for SYMBOL in ['CVX', 'IBM', 'INTC', 'MSFT', 'GS']:
        print('evaluamos : {}'.format(SYMBOL))
    #for SYMBOL in ['WMT']:
        for TYPE_DATA in TYPE_DATA_ARRAY:
            DIR_DATA = DATA_PATH + TYPE_DATA + 'train'
            x_train, y_train = predict.prepare_data(SYMBOL=SYMBOL, OUTPUT_FEATURES=OUTPUT_FEATURES, DIR_DATA=DIR_DATA)
            #print(x_train)
            print(y_train)
            # #x_test, y_test = predict.load_data_test(data.iloc[split_pos:], predict.length_of_sequences)
            model = predict.train(x_train, y_train, is_first=is_first)
            # Guardar el Modelo
            model.save(SAVE_DIR + TYPE_DATA + SYMBOL + '/' + 'lstm_{}.h5'.format(OUTPUT_TYPE), include_optimizer=True,
                       save_format='h5')
            #is_first=False


