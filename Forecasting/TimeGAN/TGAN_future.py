import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from Data.Technical_Analysis import *
from tensorflow.keras.models import Sequential, Model,save_model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input,LSTM,Dropout,Conv1D,Conv2D,LeakyReLU,BatchNormalization,AveragePooling1D,Flatten
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from Forecasting.learning_rates import *
import datetime
import statistics
#from keras.backend import print_tensor
from tensorflow.keras.backend import print_tensor

import mplfinance as mpf


import arrow

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


def batch_noise_structure(n_batch, prev, seq_len,n_seq):
    final_total = []
    for i in range(n_batch):
        for j in range(n_seq):
            random_values=np.random.normal(loc=0.0, scale=np.mean(prev[i, :, 0], dtype=np.float64), size=(seq_len, 1))
            if j==0:
                random_noise = random_values
            else:
                random_noise=np.append(random_noise,random_values,axis=1)
        final_total.append(random_noise)
    return np.array(final_total)


# params={'batch_size':8,'cnn_lr_max':1.633936,'dropout':0.120942,'strides':1,
#         'alfa_relu':0.082678,'hidden_dim':160,'cnn_filter':39,'batchnorm_momentum':0.023548,'lstm_layers':2,'lstm_neurons':306}



class TGAN():

    def __init__(self,seq_len=5,experiment=1,
                 train_step=1,save_dir='tgan',save_model=False,trainable=True):

        self.is_trainable = trainable
        self.experiment = experiment
        self.seq_len = seq_len



        if self.experiment == 1:
            self.hidden_dim = 200
            self.batch_size = 5
            self.train_steps =254

        elif self.experiment == 2:
            self.train_steps = train_step
            self.hidden_dim = 85
            self.batch_size = 2
            self.n_layers = 1
            self.gen_layers = 4
            self.dis_layers = 5
            self.sup_layers = 4

        elif self.experiment == 3:
            params_ibm= {'batch_size': 1, 'cnn_lr_max': 0.1, 'dropout': 0.01,
                                          'alfa_relu': 0.1, 'hidden_dim': 200, 'cnn_filter': 64,
                                          'batchnorm_momentum': 0.99, 'lstm_layers': 5,
                                          'lstm_neurons': 371}

            params_csco= {'batch_size': 1, 'cnn_lr_max':0.779086, 'dropout': 0.254180,
                                          'alfa_relu': 0.016624, 'hidden_dim': 470, 'cnn_filter': 61,
                                          'batchnorm_momentum': 0.965805, 'lstm_layers': 2,
                                          'lstm_neurons': 960}

            params_intc= {'batch_size': 8, 'cnn_lr_max':1.633936, 'dropout': 0.120942,
                                          'alfa_relu': 0.082678, 'hidden_dim': 160, 'cnn_filter': 39,
                                          'batchnorm_momentum': 0.023548, 'lstm_layers': 2,
                                          'lstm_neurons': 306}
            params=params_csco
            self.train_steps = train_step
            self.hidden_dim = params['hidden_dim']
            self.batch_size = params['batch_size']
            self.dropout = params['dropout']
            self.cnn_lr_max = params['cnn_lr_max']
            self.strides = 1
            self.alfa = params['alfa_relu']
            self.filter = params['cnn_filter']
            self.batch_norm = params['batchnorm_momentum']
            self.lstm_layers = params['lstm_layers']
            self.lstm_neurons = params['lstm_neurons']

        self.actual_step = 0
        self.pre_train_steps = int(self.train_steps / 2)
        # self.is_trainable=trainable
        # self.hidden_dim=hidden_dim
        # self.lstm_layers=lstm_layer
        # self.lstm_neurons=lstm_neurons
        # self.lstm_lr_max=lstm_lr_max
        # self.batchnorm_momentum=batchnorm_momentum
        # self.alfa_relu=alfa_relu
        # self.cnn_lr_max=cnn_lr_max
        # self.cnn_filter=cnn_filter
        # self.strides=strides
        # self.train_steps = train_step
        # self.eval_steps = eval_step
        # self.seq_len=seq_len
        # self.batch_size=batch_size
        # self.dropout=dropout
        # self.actual_step=0
        # self.n_seq=n_seq
        if not self.is_trainable:
            self.batch_size=30
        self.is_save=save_model
        if self.is_save:
            self.results_path = Path(save_dir)
            if not self.results_path.exists():
                self.results_path.mkdir()
            experiment = 0
            self.log_dir = self.results_path / f'experiment_{experiment:02}'
            if not self.log_dir.exists():
                self.log_dir.mkdir(parents=True)

            self.hdf_store = self.results_path / 'TimeSeriesGAN.h5'
            self.writer = tf.summary.create_file_writer(self.log_dir.as_posix())

    def prepare_data(self,SYMBOL,TYPE_INPUT,OUTPUT_FEATURES,
                     DIR_DATA ,is_separated=True, plot_data=False,is_train=True):
        #PREPARAMOS LOS DATOS
        if is_train:
            DIR_DATA = DIR_DATA + 'train'
        else:
            DIR_DATA = DIR_DATA + 'test'
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
        time_data_a = []
        self.n_seq=len(self.df.columns)
        ##Normalizan los datos
        self.scaler = MinMaxScaler()
        scaled_real = self.scaler.fit_transform(self.df).astype(np.float32)
        data_a = []
        if is_separated:
            n_win = int(len(self.df) / (self.seq_len))
            pivot = 0
            for i in range(n_win):
                data_a.append(scaled_real[pivot:pivot + self.seq_len])
                time_data_a.append(time_index[pivot:pivot + self.seq_len])
                pivot += self.seq_len
        else:
            for i in range(len(self.df) - self.seq_len):
                data_a.append(scaled_real[i:i + self.seq_len])
                time_data_a.append(time_index[i:i + self.seq_len])
        data_P1=[]
        data_X1=[]
        time_data_A=[]
        for i in range(len(data_a) - 1):
            seq_p = data_a[i]
            seq_x= data_a[i+1]
            time=np.append(time_data_a[i], time_data_a[i+1], axis=0)
            if len(seq_p) == self.seq_len and len(seq_x) == self.seq_len:
                data_X1.append(seq_x)
                data_P1.append(seq_p)
                time_data_A.append(time)
        data_X1=np.array(data_X1)
        data_P1=np.array(data_P1)
        def get_random_windows():
            while True:
                if self.is_trainable:
                    number_of_rows = data_X1.shape[0]
                    random_indices = np.random.choice(number_of_rows, size=self.batch_size, replace=False)
                    yield data_X1[random_indices, :],data_P1[random_indices, :]
                else:
                    indices =range(self.batch_size)
                    yield data_X1[indices, :],data_P1[indices, :]
        if TYPE_INPUT=='PREV_WINDOW':
            df_gen =df.copy()
            self.input_gen=len(df_gen.columns)
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
            data_P = []
            data_X = []
            data_Z= []
            for i in range(len(data_real) - 1):
                seq_z = data_z[i]
                seq_x = data_real[i]
                if len(seq_z[:self.seq_len]) == self.seq_len and len(seq_x[self.seq_len:]) == self.seq_len:
                    data_Z.append(seq_z[:self.seq_len])
                    data_X.append(seq_x[self.seq_len:])
                    data_P.append(seq_x[:self.seq_len])

            data_P = np.array(data_P)
            data_X = np.array(data_X)
            data_Z= np.array(data_Z)

            def get_random_batch():
                while True:
                    if self.is_trainable:
                        number_of_rows = data_X.shape[0]
                        random_indices = np.random.choice(number_of_rows, size=self.batch_size, replace=False)
                        yield data_X[random_indices, :], data_Z[random_indices, :], data_P[random_indices, :]
                    else:
                        #indices=np.linspace(0,self.batch_size,self.batch_size-1)
                        indices=range(len(data_P))
                        #print(indices)
                        yield data_X[indices, :], data_Z[indices, :], data_P[indices, :]
            self.time=time_data_real
            both_series = (tf.data.Dataset.from_generator(get_random_batch, output_types=(tf.float32, tf.float32,tf.float32)))
            self.both_series = iter(both_series)
        elif TYPE_INPUT=='RANDOM':
            self.input_gen=self.n_seq
            def make_random_data():
                while True:
                    yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.input_gen))

            # entrada con valores con distribucion uniforme
            self.random_series = iter(tf.data.Dataset
                                      .from_generator(make_random_data, output_types=tf.float32).batch(self.batch_size)
                                      .repeat())
            ##Data real
            #x_series = (tf.data.Dataset.from_generator(get_random_windows,output_types=(tf.float32,tf.float32)))
            #self.real_series = iter(x_series)
            self.time=time_data_A

        # def make_noise_data():
        #     while True:
        #         prev_data = next(prev_series_iter)
        #         prev = np.array(prev_data)
        #         random_final = batch_noise_structure(n_batch=len(prev), prev=prev, seq_len=self.seq_len,n_seq=self.n_seq)
        #         #print(random_final)
        #         noised_data = prev_data + random_final
        #         #noised_data=prev_data + noised_data
        #         yield noised_data


        self.individual_series=iter(tf.data.Dataset.from_generator(get_random_windows,output_types=(tf.float32,tf.float32)))



        ##Entradas de generador
        ##entradas de ventanas previas sin ruido
        #z_series = (tf.data.Dataset.from_tensor_slices(data_Z_2).batch(self.batch_size))
        #self.random_series = iter(z_series.repeat())







    def create_model(self):
        ##create_model
        self.X = Input(shape=[self.seq_len, self.n_seq], name='RealData')
        self.Z = Input(shape=[self.seq_len,self.input_gen], name='RandomData')

        # def make_GRU(n_layers, hidden_units, output_units,name):
        #     return Sequential([GRU(units=hidden_units,
        #                            return_sequences=True,
        #                            name=f'GRU_{i + 1}') for i in range(n_layers)] +
        #                       [Dense(units=output_units,
        #                              activation='sigmoid',
        #                              name='OUT')], name=name)
        def make_LSTM_gen(n_layers, hidden_units, output_units,dropout,norm, name):
            return Sequential([LSTM(units=hidden_units,
                                   return_sequences=True,dropout=dropout,
                                   name=f'LSTM_{i + 1}') for i in range(n_layers)] +
                              #[Dropout(rate=dropout, name='Dropout')] +
                              [BatchNormalization(momentum=norm)] +
                              [Dense(units=output_units,
                                     name='OUT')], name=name)

        # def make_LSTM_dis(n_layers, hidden_units, output_units, name):
        #     return Sequential([LSTM(units=hidden_units,
        #                            return_sequences=True,
        #                            name=f'LSTM_{i + 1}') for i in range(n_layers)] +
        #                       [LeakyReLU(alpha=0.01)] +
        #                       [BatchNormalization(momentum=0.99)] +
        #                       [Dense(units=output_units,activation='sigmoid',
        #                              name='OUT')], name=name)

        def make_LSTM(n_layers, hidden_units, output_units,activation_output, name):
            return Sequential([LSTM(units=hidden_units,
                                   return_sequences=True,
                                   name=f'LSTM_{i + 1}') for i in range(n_layers)] +
                              [Dense(units=output_units,
                                     activation=activation_output,
                                     name='OUT')], name=name)

        def make_CNN(output_units, name,alpha,mom,initial_filter):
            input_shape = (self.batch_size, self.seq_len,self.hidden_dim)
            input_layer = Input(input_shape[1:])

            if input_shape[0] < 60:  # for italypowerondemand dataset
                padding = 'same'
            conv1 = Conv1D(filters=initial_filter, kernel_size=5, padding='same',strides=self.strides)(input_layer)
            #conv1 = AveragePooling1D(pool_size=3)(conv1)
            conv1=LeakyReLU(alpha=alpha)(conv1)
            conv2 = Conv1D(filters=2*initial_filter, kernel_size=5, padding='same',strides=self.strides)(conv1)
            conv2 = LeakyReLU(alpha=alpha)(conv2)
            conv2=BatchNormalization(momentum=mom)(conv2)
            conv3 = Conv1D(filters=4*initial_filter, kernel_size=5, padding='same',strides=self.strides)(conv2)
            conv3 = LeakyReLU(alpha=alpha)(conv3)
            conv3 = BatchNormalization(momentum=mom)(conv3)
            # conv2 = AveragePooling1D(pool_size=3)(conv2)
            #conv2 = AveragePooling1D(pool_size=3)(conv2)
            #flatten_layer = Flatten()(conv3)
            output_layer = Dense(units=output_units, activation='sigmoid')(conv3)
            model = Model(inputs=input_layer, outputs=output_layer,name=name)
            return model

        if self.experiment==1:

            self.embedder = make_LSTM(n_layers=3,
                                 hidden_units=self.hidden_dim,
                                 output_units=self.hidden_dim,activation_output='sigmoid',
                                 name='Embedder')

            self.recovery = make_LSTM(n_layers=3,
                                 hidden_units=self.hidden_dim,
                                 output_units=self.n_seq,activation_output='sigmoid',
                                 name='Recovery')

            self.generator = make_LSTM(n_layers=3,
                                       hidden_units=self.hidden_dim,
                                       output_units=self.hidden_dim, activation_output='sigmoid',
                                       name='Generator')

            self.discriminator = make_LSTM(n_layers=3,
                                           hidden_units=self.hidden_dim,
                                           output_units=1, activation_output='sigmoid',
                                           name='Discriminator')

            self.supervisor = make_LSTM(n_layers=2,
                                        hidden_units=self.hidden_dim,
                                        output_units=self.hidden_dim, activation_output='sigmoid',
                                        name='Supervisor')
            self.autoencoder_optimizer = Adam()
            ##Optimizadores
            self.generator_optimizer = Adam()
            self.discriminator_optimizer = Adam()
            self.embedding_optimizer = Adam()

            self.supervisor_optimizer = Adam()
        elif self.experiment==2:
            self.embedder = make_LSTM(n_layers=self.n_layers,
                                 hidden_units=self.hidden_dim,
                                 output_units=self.hidden_dim,activation_output='sigmoid',
                                 name='Embedder')

            self.recovery = make_LSTM(n_layers=self.n_layers,
                                 hidden_units=self.hidden_dim,
                                 output_units=self.n_seq,activation_output='sigmoid',
                                 name='Recovery')

            self.generator = make_LSTM(n_layers=self.gen_layers,
                                       hidden_units=self.hidden_dim,
                                       output_units=self.hidden_dim, activation_output='sigmoid',
                                       name='Generator')

            self.discriminator = make_LSTM(n_layers=self.dis_layers,
                                           hidden_units=self.hidden_dim,
                                           output_units=1, activation_output='sigmoid',
                                           name='Discriminator')

            self.supervisor = make_LSTM(n_layers=self.sup_layers,
                                        hidden_units=self.hidden_dim,
                                        output_units=self.hidden_dim, activation_output='sigmoid',
                                        name='Supervisor')
            self.autoencoder_optimizer = Adam()
            ##Optimizadores
            self.generator_optimizer = Adam()
            self.discriminator_optimizer = Adam()
            self.embedding_optimizer = Adam()
            self.supervisor_optimizer = Adam()

        elif self.experiment==3:
            self.embedder = make_LSTM(n_layers=3,
                                      hidden_units=self.hidden_dim,
                                      output_units=self.hidden_dim, activation_output='sigmoid',
                                      name='Embedder')

            self.recovery = make_LSTM(n_layers=3,
                                      hidden_units=self.hidden_dim,
                                      output_units=self.n_seq, activation_output='sigmoid',
                                      name='Recovery')

            self.generator = make_LSTM_gen(n_layers=self.lstm_layers,
                                  hidden_units=self.lstm_neurons,
                                  output_units=self.hidden_dim,dropout=self.dropout,norm=self.batch_norm,
                                  name='Generator')
            self.discriminator = make_CNN(output_units=1,alpha=self.alfa,mom=self.batch_norm,initial_filter=self.filter,
                                         name='Discriminator')

            self.supervisor = make_LSTM(n_layers=2,
                                        hidden_units=self.hidden_dim,
                                        output_units=self.hidden_dim, activation_output='sigmoid',
                                        name='Supervisor')

            ##Autoencoder Optimization
            # starter_learning_rate = 0.1
            # end_learning_rate = 0.01
            cycle_lenght = self.train_steps / 3
            # steps_decay = self.train_steps
            schedule_gen = CyclicalSchedule(TriangularSchedule, min_lr=0.001, max_lr=0.1, cycle_length=cycle_lenght)
            gen_lr_fn = schedule_gen(self.actual_step)
            schedule_dis = CyclicalSchedule(TriangularSchedule, min_lr=0.001, max_lr=self.cnn_lr_max,
                                            cycle_length=cycle_lenght)
            dis_lr_fn = schedule_dis(self.actual_step)
            schedule_encoder = CyclicalSchedule(TriangularSchedule, min_lr=0.001, max_lr=0.1, cycle_length=cycle_lenght)
            encoder_lr_fn = schedule_encoder(self.actual_step)

            # lr_fn_autoencoder = tf.optimizers.schedules.PolynomialDecay(initial_learning_rate=starter_learning_rate, decay_steps=steps_decay, end_learning_rate=end_learning_rate)
            self.autoencoder_optimizer = Adam(encoder_lr_fn)
            ##Optimizadores
            # lr_fn_generator = tf.optimizers.schedules.PolynomialDecay(initial_learning_rate=starter_learning_rate, decay_steps=steps_decay, end_learning_rate=end_learning_rate)
            self.generator_optimizer = Adam(gen_lr_fn)
            # lr_fn_discriminator = tf.optimizers.schedules.PolynomialDecay(initial_learning_rate=starter_learning_rate, decay_steps=steps_decay, end_learning_rate=end_learning_rate)
            self.discriminator_optimizer = Adam(dis_lr_fn)
            # lr_fn_embedding = tf.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.5, decay_steps=steps_decay, end_learning_rate=0.01)
            self.embedding_optimizer = Adam(encoder_lr_fn)
            # lr_fn_supervisor = tf.optimizers.schedules.PolynomialDecay(initial_learning_rate=starter_learning_rate, decay_steps=steps_decay, end_learning_rate=end_learning_rate)
            self.supervisor_optimizer = Adam(encoder_lr_fn)

        # self.embedder = make_LSTM(n_layers=3,
        #                          hidden_units=self.hidden_dim,
        #                          output_units=self.hidden_dim,activation_output='sigmoid',
        #                          name='Embedder')
        #
        # self.recovery = make_LSTM(n_layers=3,
        #                          hidden_units=self.hidden_dim,
        #                          output_units=self.n_seq,activation_output='sigmoid',
        #                          name='Recovery')
        #
        # self.generator = make_LSTM_gen(n_layers=self.lstm_layers,
        #                           hidden_units=self.lstm_neurons,norm=self.batchnorm_momentum,
        #                           output_units=self.hidden_dim,dropout=self.dropout,
        #                           name='Generator')
        #
        # # self.generator = make_LSTM(n_layers=3,
        # #                           hidden_units=452,
        # #                           output_units=self.hidden_dim,activation_output='sigmoid',
        # #                           name='Generator')
        #
        # # self.discriminator = make_LSTM(n_layers=3,
        # #                               hidden_units=self.hidden_dim,
        # #                               output_units=1,activation_output='sigmoid',
        # #                               name='Discriminator')
        #
        # # self.discriminator = make_LSTM_dis(n_layers=3,
        # #                               hidden_units=self.hidden_dim,
        # #                               output_units=1,
        # #                               name='Discriminator')
        #
        #
        # self.discriminator = make_CNN(output_units=1,alpha=self.alfa_relu,mom=self.batchnorm_momentum,initial_filter=self.cnn_filter,
        #                               name='Discriminator')
        # self.supervisor = make_LSTM(n_layers=2,
        #                            hidden_units=self.hidden_dim,
        #                            output_units=self.hidden_dim,activation_output='sigmoid',
        #                            name='Supervisor')

        self.gamma = 1

        ##Generic loss functions
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()

        ##Fase 1 Entrenamiento de los autoencoder
        self.H = self.embedder(self.X)
        X_tilde = self.recovery(self.H)

        self.autoencoder = Model(inputs=self.X,
                                 outputs=X_tilde,
                                 name='Autoencoder')

    def training_loop(self,SYMBOL,TYPE_INPUT,OUTPUT_FEATURES,DIR_DATA,is_separated, plot_data):

        if SYMBOL==None:
            name_stocks = ['AAPL', 'CSCO', 'CVX', 'IBM', 'INTC','MSFT','GS']
            for stock in name_stocks:
                ##Prepare data
                self.prepare_data(SYMBOL=stock,TYPE_INPUT=TYPE_INPUT,OUTPUT_FEATURES=OUTPUT_FEATURES,DIR_DATA=DIR_DATA,
                                  is_separated=is_separated, plot_data=plot_data)
                ##Create_model
                self.create_model()
                # print('paso el prepare data')
                ##Training Encoder
                self.training_autoencoder()
                # print('paso el training autoencoder')
                ###Fase 2 Entrenamiento supervisado
                self.training_supervisor()
                # print('training supervisor')
                ###joint training
                self.joint_training(TYPE_INPUT=TYPE_INPUT,total_steps=self.train_steps)
                # print('paso el joint training')
        else:
            ##Prepare data
            self.prepare_data(SYMBOL=SYMBOL, TYPE_INPUT=TYPE_INPUT, OUTPUT_FEATURES=OUTPUT_FEATURES, DIR_DATA=DIR_DATA,
                              is_separated=is_separated, plot_data=plot_data,is_train=True)
            self.create_model()
            # print('paso el prepare data')
            ##Training Encoder
            self.training_autoencoder()
            # print('paso el training autoencoder')
            ###Fase 2 Entrenamiento supervisado
            self.training_supervisor()
            ###joint training
            self.train_history = pd.DataFrame()
            self.joint_training(TYPE_INPUT=TYPE_INPUT,total_steps=self.train_steps,history=self.train_history)
            # print(' train history :')
            # print(self.train_history)
            # self.train_history.plot()
            # plt.show()
            # print('Termina el proceso de entrenamiento:')
            #
            # self.prepare_data(SYMBOL=SYMBOL, TYPE_INPUT=TYPE_INPUT, OUTPUT_FEATURES=OUTPUT_FEATURES, DIR_DATA=DIR_DATA,
            #                   is_separated=is_separated, plot_data=plot_data,is_train=False)
            #
            # print('Comienza el proceso de evaluacion:')
            # self.eval_history=pd.DataFrame()
            # self.joint_training(TYPE_INPUT=TYPE_INPUT, total_steps=self.eval_steps,history=self.eval_history)
            # print(' eval history :')
            # print(self.eval_history)
            # if plot_data:
            #     self.train_history.plot()
            #     self.eval_history.plot()
            #     plt.show()







    def training_autoencoder(self):

        ###Autoencoder training steps
        @tf.function
        def train_autoencoder_init(x):
            with tf.GradientTape() as tape:
                x_tilde = self.autoencoder(x)
                embedding_loss_t0 = self.mse(x, x_tilde)
                e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

            var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
            gradients = tape.gradient(e_loss_0, var_list)
            self.autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)

        ##Entrenamiento del Autoencoder
        for step in tqdm(range(self.train_steps)):
            X_,P_ = next(self.individual_series)
            #print_tensor(X_,message='entrenamiento del autoencoder')
            if step % 2 == 0:
                step_e_loss_t0 = train_autoencoder_init(X_)
            else:
                step_e_loss_t0 = train_autoencoder_init(P_)
            if self.is_save:
                with self.writer.as_default():
                    tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)
        if self.is_save:
            self.autoencoder.save(self.log_dir /'autoencoder.h5',include_optimizer=True,save_format='h5')




    def training_supervisor(self):
        # funcion de entrenamiento del supervisor
        @tf.function
        def train_supervisor(x):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                h_hat_supervised = self.supervisor(h)
                #g_loss_s = self.loss_function(h[:, 1:, :], h_hat_supervised[:, 1:, :])
                g_loss_s = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            var_list = self.supervisor.trainable_variables
            gradients = tape.gradient(g_loss_s, var_list)
            self.supervisor_optimizer.apply_gradients(zip(gradients, var_list))
            return g_loss_s

        ##Loop de entrenamiento del supervisor
        for step in tqdm(range(self.train_steps)):
            X_,P_ = next(self.individual_series)
            #step_g_loss_s = train_supervisor(X_)
            if step % 2 == 0:
                step_g_loss_s = train_supervisor(X_)
            else:
                step_g_loss_s = train_supervisor(P_)
            if self.is_save:
                with self.writer.as_default():
                    tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=step)
        if self.is_save:
            self.supervisor.save(self.log_dir / 'supervisor.h5',include_optimizer=True,save_format='h5')


    def joint_training(self,TYPE_INPUT,total_steps,history):
        # Entramiento conjunto
        E_hat = self.generator(self.Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)

        adversarial_supervised = Model(inputs=self.Z,
                                       outputs=Y_fake,
                                       name='AdversarialNetSupervised')

        adversarial_supervised.summary()

        # Arquitectura adversaria en el espacio latente
        Y_fake_e = self.discriminator(E_hat)

        adversarial_emb = Model(inputs=self.Z,
                                outputs=Y_fake_e,
                                name='AdversarialNet')

        ##promedio y varianza de coste
        X_hat = self.recovery(H_hat)
        self.synthetic_data = Model(inputs=self.Z,
                               outputs=X_hat,
                               name='SyntheticData')

        def get_generator_moment_loss(y_true, y_pred):
            y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
            y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
            g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
            g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
            return g_loss_mean + g_loss_var

        def calculate_accuracy(y_true, y_pred):
            real = y_true + 1
            predict = y_pred + 1
            cuadrado = tf.square((real - predict) / real)
            axis = list(range(len(cuadrado.get_shape()) - 1))
            mean, var = tf.nn.moments(x=cuadrado, axes=axis)
            raiz = tf.reduce_mean(tf.sqrt(mean))
            percentage = 1 - raiz
            return percentage

        # Discriminador
        # Arquitectura: data real
        Y_real = self.discriminator(self.H)
        discriminator_model = Model(inputs=self.X,
                                    outputs=Y_real,
                                    name='DiscriminatorReal')

        @tf.function
        def train_generator(x, z):
            with tf.GradientTape() as tape:
                y_fake = adversarial_supervised(z)
                generator_loss_unsupervised = self.bce(y_true=tf.ones_like(y_fake),
                                                  y_pred=y_fake)

                y_fake_e = adversarial_emb(z)
                generator_loss_unsupervised_e = self.bce(y_true=tf.ones_like(y_fake_e),
                                                    y_pred=y_fake_e)
                h = self.embedder(x)
                h_hat_supervised = self.supervisor(h)
                generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

                x_hat = self.synthetic_data(z)
                generator_moment_loss = get_generator_moment_loss(x, x_hat)

                generator_acurracy=calculate_accuracy(y_true=x,y_pred=x_hat)

                generator_loss = (generator_loss_unsupervised +
                                  generator_loss_unsupervised_e +
                                  100 * tf.sqrt(generator_loss_supervised) +
                                  100 * generator_moment_loss)

            var_list = self.generator.trainable_variables + self.supervisor.trainable_variables
            gradients = tape.gradient(generator_loss, var_list)
            self.generator_optimizer.apply_gradients(zip(gradients, var_list))
            return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss,generator_acurracy

        @tf.function
        def train_embedder(x):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                h_hat_supervised = self.supervisor(h)
                generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

                x_tilde = self.autoencoder(x)
                embedding_loss_t0 = self.mse(x, x_tilde)
                e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

            var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
            gradients = tape.gradient(e_loss, var_list)
            self.embedding_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)

        @tf.function
        def get_discriminator_loss(x, z):
            y_real = discriminator_model(x)
            discriminator_loss_real = self.bce(y_true=tf.ones_like(y_real),
                                          y_pred=y_real)

            y_fake = adversarial_supervised(z)
            discriminator_loss_fake = self.bce(y_true=tf.zeros_like(y_fake),
                                          y_pred=y_fake)

            y_fake_e = adversarial_emb(z)
            discriminator_loss_fake_e = self.bce(y_true=tf.zeros_like(y_fake_e),
                                            y_pred=y_fake_e)
            return (discriminator_loss_real +
                    discriminator_loss_fake +
                    self.gamma * discriminator_loss_fake_e)

        @tf.function
        def train_discriminator(x, z):
            with tf.GradientTape() as tape:
                discriminator_loss = get_discriminator_loss(x, z)

            var_list = self.discriminator.trainable_variables
            gradients = tape.gradient(discriminator_loss, var_list)
            self.discriminator_optimizer.apply_gradients(zip(gradients, var_list))
            return discriminator_loss

        # Training LOOP
        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_g_acc = total_acc = 0
        d_loss_array,g_loss_u_array ,g_loss_s_array ,g_loss_v_array,e_loss_t0_array ,g_acc_array ,acc_array=[],[],[],[],[],[],[]
        for step in range(total_steps):
            # Train generator (twice as often as discriminator)
            self.actual_step=step
            for kk in range(2):
                if TYPE_INPUT=='RANDOM':
                    Z_ = next(self.random_series)

                    if step % 2== 0:
                        X_,P_ = next(self.individual_series)
                    else:
                        P_,X_=next(self.individual_series)
                else:
                    X_,Z_,P_= next(self.both_series)
                # Train generator
                step_g_loss_u, step_g_loss_s, step_g_loss_v,step_g_acc = train_generator(X_, Z_)

                # Train embedder
                step_e_loss_t0 = train_embedder(X_)

            if TYPE_INPUT == 'RANDOM':
                Z_ = next(self.random_series)
                if step % 2 == 0:
                    X_, P_ = next(self.individual_series)
                else:
                    P_, X_ = next(self.individual_series)
            else:
                X_, Z_, P_= next(self.both_series)
            step_d_loss = get_discriminator_loss(X_, Z_)
            if step_d_loss > 1.15 :
                #print('se entrena el discriminador')
                step_d_loss = train_discriminator(X_, Z_)


            d_loss_array.append(step_d_loss.numpy())
            g_loss_u_array.append(step_g_loss_u.numpy())
            g_loss_v_array.append(step_g_loss_v.numpy())
            g_loss_s_array.append(step_g_loss_s.numpy())
            e_loss_t0_array.append(step_e_loss_t0.numpy())
            acc_array.append(step_g_acc.numpy())
            if step % 2 == 0:

                print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                      f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | g_acc: {step_g_acc:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')


            if self.is_save:
                with self.writer.as_default():
                    tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
                    tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
                    tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
                    tf.summary.scalar('Acc', step_g_acc, step=step)
                    tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
                    tf.summary.scalar('D Loss', step_d_loss, step=step)
            acc = statistics.mean(acc_array)
            if (acc)>=0.99 and self.actual_step>=100:
                print(' accuracy total es :')
                print(acc)
                print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                      f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | g_acc: {step_g_acc:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')
                print('la diferencia entre los costes es :')
                print(step_d_loss - step_g_loss_u)
                #break

        history['d_loss']=d_loss_array
        history['g_loss_u']=g_loss_u_array
        history['g_loss_v']=g_loss_v_array
        history['g_acc']=acc_array
        history['e_loss_t0']=e_loss_t0_array
        if self.is_save:
            self.synthetic_data.save(self.log_dir / 'synthetic_data.h5',include_optimizer=True,save_format='h5')

    def plot_data(self):

        #generated_data = []
        #for i in range(int(self.n_windows / self.batch_size)):
        Z_ = next(self.random_series)
        d = self.synthetic_data(Z_)
        R_ = next(self.real_series_iter)
        print('Este es el real')
        #R_=R_[:,:,:5]
        print(R_)
        real_data = np.array(R_)
        print(real_data.shape)
        #d=d[:,:,:5]
        generated_data = np.array(d)
        print(generated_data.shape)
        print(len(generated_data))
        generated_data = (self.scaler.inverse_transform(generated_data.reshape(-1, self.n_seq)).reshape(-1, self.seq_len, self.n_seq))
        real_data = (self.scaler.inverse_transform(real_data.reshape(-1, self.n_seq)).reshape(-1, self.seq_len, self.n_seq))

        for i in range(3):
            # print(' sequence:{}'.format(i))
            generated = generated_data[i, :, :5]
            real = real_data[i, :, :5]
            # print(generated)
            # numpy_data = np.array([[1, 2], [3, 4]])
            base = datetime.datetime(2020, 11, 20)
            arr = np.array([base + datetime.timedelta(days=j) for j in range(20)])
            didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
            gen_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
            print(gen_df)
            real_df = pd.DataFrame(data=real, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
            # mpf.plot(real_dat, type='candle',mav=(3,6,9),volume=True)
            fig = mpf.figure(style='yahoo', figsize=(14, 16))

            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)

            av1 = fig.add_subplot(3, 2, 5, sharex=ax1)
            av2 = fig.add_subplot(3, 2, 6, sharex=ax2)

            mpf.plot(gen_df, type='candle', ax=ax1, volume=av1, mav=(10, 20), axtitle='generated data')
            mpf.plot(real_df, type='candle', ax=ax2, volume=av2, mav=(10, 20), axtitle='real data')

            mpf.show()

