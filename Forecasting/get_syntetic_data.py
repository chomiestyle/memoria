import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
#from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from Forecasting.TimeGAN.TGAN_future import *
from yahoo_fin import *
from yahoo_fin.stock_info import *
import arrow
from Forecasting.LSTM.LSTM import calculate_accuracy
from Data.Technical_Analysis import get_technical_indicator
from Forecasting.Data_Visualization import *



def anchor(signal, weight):
	buffer = []
	last = signal[0]
	for i in signal:
		smoothed_val = last * weight + (1 - weight) * i
		buffer.append(smoothed_val)
		last = smoothed_val
	return buffer




def get_sintetic_model(saved_path):
    #Recrea exactamente el mismo modelo solo desde el archivo
    new_model =load_model(saved_path)
    #new_model=load()
    return new_model

import numpy as np
import datetime


def seq_sinthetic_data(n,synthetic_data,z_iter):
    array=[]
    for i in range(n):
        Z_ = next(z_iter)
        d_1 = synthetic_data(Z_)
        generated_data_1 = np.array(d_1)
        array.append(generated_data_1)
    total_data=array[0]
    for data in array[1:]:
        total_data=total_data+data
    mean_data=total_data/len(array)
    return mean_data

def fib_sinthetic_data(n,synthetic_model,input):
    for i in range(n):
        input=synthetic_model(input)

    return np.array(input)

def get_syntetic_data(gen_model,
                      lstm_model_1,lstm_model_2,TYPE_INPUT,OUTPUT_FEATURES,DATA_DIR,symbol,is_pred,is_gen):
    #Inicializamos las variables de salida
    gen_data, real_data, prev_data, pred_data,pred_data2, t_data=[],[],[],[],[],[]
    #Creamos un objeto TGAN para obtener la data
    timegan=TGAN(seq_len=5,trainable=False)
    timegan.prepare_data(SYMBOL=symbol,TYPE_INPUT=TYPE_INPUT,
                              OUTPUT_FEATURES=OUTPUT_FEATURES,
                              DIR_DATA=DATA_DIR,
                              is_separated=False,is_train=False)

    #Preparamos segun el tipo de data
    if TYPE_INPUT=='PREV_WINDOW':
        data_iter=timegan.both_series
        R_, Z_, P_ = next(data_iter)
    elif TYPE_INPUT=='RANDOM':
        z_iter = timegan.random_series
        data_iter = timegan.individual_series
        R_, P_=next(data_iter)
        Z_=next(z_iter)
    t_data=timegan.time
    prev_data=np.array(P_)
    real_data=np.array(R_)
    real_data = (timegan.scaler.inverse_transform(real_data.reshape(-1,timegan.n_seq)).reshape(-1, timegan.seq_len,timegan.n_seq))
    prev_data = (timegan.scaler.inverse_transform(prev_data.reshape(-1,timegan.n_seq)).reshape(-1, timegan.seq_len, timegan.n_seq))
    real_data=join_data(future_data=real_data,prev_data=prev_data,is_adjust=False)

    if is_pred:
        #Obtenemos los valores del LSTM
        if type(lstm_model_1) == str and type(lstm_model_2) == str:
            lstm_model_1= load_model(lstm_model_1,custom_objects={'calculate_accuracy': calculate_accuracy})
            lstm_model_1.summary()
            lstm_model_2 = load_model(lstm_model_2, custom_objects={'calculate_accuracy': calculate_accuracy})
        print(P_)
        predicted_data_1 = lstm_model_1.predict(P_)
        pred_data = np.array(predicted_data_1)
        pred_data = (timegan.scaler.inverse_transform(pred_data.reshape(-1, timegan.n_seq)).reshape(-1, timegan.seq_len, timegan.n_seq))
        pred_data=join_data(future_data=pred_data,prev_data=prev_data,is_adjust=False)
        predicted_data_2 = lstm_model_2.predict(Z_)
        pred_data2 = np.array(predicted_data_2)
        pred_data2 = (timegan.scaler.inverse_transform(pred_data2.reshape(-1, timegan.n_seq)).reshape(-1, timegan.seq_len, timegan.n_seq))
        pred_data2=join_data(future_data=pred_data2,prev_data=prev_data,is_adjust=False)
    if is_gen:
        #Obtenemos los valores de la TGAN
        if type(gen_model) == str:
            gen_model=get_sintetic_model(saved_path=gen_model)
        generated_data=gen_model.predict(Z_)
        gen_data = np.array(generated_data)
        gen_data = (timegan.scaler.inverse_transform(gen_data.reshape(-1, timegan.n_seq)).reshape(-1, timegan.seq_len, timegan.n_seq))
        gen_data = join_data(future_data=gen_data,prev_data=prev_data,is_adjust=False)
    #generated_data=seq_sinthetic_data(n=20,synthetic_data=synthetic_data_1,z_iter=z_iter)
    #generated_data=fib_sinthetic_data(n=9,synthetic_model=gen_model,input=Z_)

    return gen_data,real_data,pred_data,pred_data2,t_data




def get_syntetic_data_tgan_global(gen_model_dir,TYPE_INPUT,OUTPUT_FEATURES,DATA_DIR,SYMBOL,OUTPUT_TYPE):
    #Inicializamos las variables de salida
    full_data=pd.DataFrame()
    #Creamos un objeto TGAN para obtener la data
    timegan=TGAN(seq_len=5,trainable=False)
    timegan.prepare_data(SYMBOL=SYMBOL,TYPE_INPUT=TYPE_INPUT,
                              OUTPUT_FEATURES=OUTPUT_FEATURES,
                              DIR_DATA=DATA_DIR,
                              is_separated=True,is_train=False)

    #Preparamos segun el tipo de data
    if TYPE_INPUT=='PREV_WINDOW':
        data_iter=timegan.both_series
        R_, Z_, P_ = next(data_iter)
    elif TYPE_INPUT=='RANDOM':
        z_iter = timegan.random_series
        data_iter = timegan.individual_series
        R_, P_=next(data_iter)
        Z_=next(z_iter)
    t_data=timegan.time
    prev_data=np.array(P_)
    real_data=np.array(R_)
    real_data = (timegan.scaler.inverse_transform(real_data.reshape(-1,timegan.n_seq)).reshape(-1, timegan.seq_len,timegan.n_seq))
    prev_data = (timegan.scaler.inverse_transform(prev_data.reshape(-1,timegan.n_seq)).reshape(-1, timegan.seq_len, timegan.n_seq))
    real_data=join_data(future_data=real_data,prev_data=prev_data,is_adjust=False)
    dataframe_real = get_dataframe(real_data, t_data, is_anchor=False, FEATURES=['Real'])
    n_gan_epochs=[600, 900, 1000, 1200, 1500, 1800, 2000]
    #n_gan_epochs = [1800,2000]
    for n_gan_epoch in n_gan_epochs:
        gen_model_path=gen_model_dir+'/' + SYMBOL + '/' + 'TimeGAN_{}_{}_{}'.format(TYPE_INPUT, OUTPUT_TYPE, n_gan_epoch)
        gen_model= gen_model_path + '/experiment_00/synthetic_data.h5'
        gen_model=get_sintetic_model(saved_path=gen_model)
        generated_data=gen_model.predict(Z_)
        gen_data = np.array(generated_data)
        gen_data = (timegan.scaler.inverse_transform(gen_data.reshape(-1, timegan.n_seq)).reshape(-1, timegan.seq_len, timegan.n_seq))
        gen_data = join_data(future_data=gen_data,prev_data=prev_data,is_adjust=False)
        dataframe=get_dataframe(gen_data,t_data,is_anchor=False,FEATURES=['TGAN_{}'.format(n_gan_epoch)])
        full_data['TGAN_{}'.format(n_gan_epoch)] =dataframe['TGAN_{}'.format(n_gan_epoch)]


    dataframe_real['mean']=full_data.mean(axis=1).values
    print(dataframe_real)
    dataframe_real.tail(30).plot()
    plt.show()
    full_data['Real']=dataframe_real['Real']
    print(full_data)
    full_data.tail(30).plot()
    plt.show()
    return full_data,dataframe_real


def get_dataframe(data,t_data,is_anchor,FEATURES=["Open","High","Low","Close"],is_plot=False):
    full_df = pd.DataFrame()
    for i in range(len(data)):
        generated=data[i]
        arr = np.array(t_data[i])
        didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
        df = pd.DataFrame(data=generated,index=didx ,columns=FEATURES)
        if is_anchor:
            for c in df.columns:
                df[c] = anchor(df[c].values, 0.2)
        if is_plot:
            df.plot()
            plt.show()
        full_df=full_df.append(df)
    if is_anchor:
        for c in full_df.columns:
            full_df[c] = anchor(full_df[c].values, 0.4)
    return full_df

def get_dataframe_for_compare(data,data2,t_data,real_data,is_anchor,FEATURES=["Open","High","Low","Close"],is_plot=True):
    full_gen_df = pd.DataFrame()
    full_real_df=pd.DataFrame()
    for i in range(len(data)):
        generated=data[i]
        generated2 = data2[i]
        real=real_data[i]
        arr = np.array(t_data[i])
        didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
        df = pd.DataFrame(data=generated,index=didx ,columns=FEATURES)
        df2 = pd.DataFrame(data=generated2, index=didx, columns=FEATURES)
        if is_anchor:
            for c in df.columns:
                df[c] = anchor(df[c].values, 0.2)
        df_real=pd.DataFrame(data=real,index=didx ,columns=FEATURES)
        if is_plot:
            if len(FEATURES)>1:
                plot_OHLC(gen_full_df=df,real_full_df=df_real)
            else:
                plot_CLOSE_lstm(gen_full_df=df,gen_full_df2=df2,real_full_df=df_real)
        full_gen_df=full_gen_df.append(df)
        full_real_df=full_real_df.append(df_real)
    if is_anchor:
        for c in full_gen_df.columns:
            full_gen_df[c] = anchor(full_gen_df[c].values, 0.4)
    return full_gen_df,full_real_df


def get_dataframe_for_compare2(data,t_data,real_data,experiment,is_anchor,FEATURES=["Open","High","Low","Close"],is_plot=True):
    full_gen_df = pd.DataFrame()
    full_real_df=pd.DataFrame()
    for i in range(len(data)):
        generated=data[i]
        #generated2 = data2[i]
        real=real_data[i]
        arr = np.array(t_data[i])
        didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
        df = pd.DataFrame(data=generated,index=didx ,columns=FEATURES)
        #df2 = pd.DataFrame(data=generated2, index=didx, columns=FEATURES)
        if is_anchor:
            for c in df.columns:
                df[c] = anchor(df[c].values, 0.2)
        df_real=pd.DataFrame(data=real,index=didx ,columns=FEATURES)
        if is_plot:
            if len(FEATURES)>1:
                plot_OHLC(gen_full_df=df,real_full_df=df_real)
            else:
                plot_CLOSE_TGAN(gen_full_df=df,real_full_df=df_real,experiment=experiment)
        full_gen_df=full_gen_df.append(df)
        full_real_df=full_real_df.append(df_real)
    if is_anchor:
        for c in full_gen_df.columns:
            full_gen_df[c] = anchor(full_gen_df[c].values, 0.4)
    return full_gen_df,full_real_df

def join_data(future_data,prev_data,is_adjust):
    final_data=[]
    for i in range(len(future_data)):
        generated = future_data[i, :, :]
        prev = prev_data[i, :, :]
        if is_adjust:
            print('generated 1:')
            print(generated)

            ###suma delta
            first_gen = generated[0]
            print('first gen:')
            print(first_gen)
            last_prev = prev[-1]
            print('last prev:')
            print(last_prev)
            delta=first_gen-last_prev
            print('delta:')
            print(delta)
            generated=generated+delta
            print('generated 2:')
            print(generated)
        generated = np.append(prev, generated, axis=0)
        final_data.append(generated)
    return final_data



def prueba():
    #Prepare training data with future information
    DATA_PATH='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Data/combined/'

    # #TYPE_DATA='Most_Important_Features/'
    # #TYPE_DATA='Full_data'
    # #TYPE_DATA='data_assets_TI'
    TYPE_DATA = 'OHLC_NEWS/'
    TYPE_DATA_LSTM = 'data_assets_TI/'
    #
    #
    SYMBOL='INTC'
    #OUTPUT_TYPE='OHLC'
    OUTPUT_TYPE='CLOSE'
    #
    SAVE_DIR_LSTM='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/TRAIN/'
    LSTM_DIR=SAVE_DIR_LSTM+TYPE_DATA_LSTM+'/'+SYMBOL+'/'+'lstm_{}.h5'.format(OUTPUT_TYPE)
    lstm_model=LSTM_DIR
    LSTM_DIR2=SAVE_DIR_LSTM+TYPE_DATA+'/'+SYMBOL+'/'+'lstm_{}.h5'.format(OUTPUT_TYPE)
    lstm_model2=LSTM_DIR2
    OUTPUT_FEATURES=['close']
    # #GAN
    SAVE_DIR_GAN='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/TimeGAN/TRAIN/'
    #
    # #experimento 1
    # #TYPE_INPUT_exp1='RANDOM'
    # TYPE_INPUT_exp1='PREV_WINDOW'
    # TYPE_DATA_1 = 'data_assets_TI/'
    # DIR_DATA=DATA_PATH+TYPE_DATA
    # TGAN_DIR_1=SAVE_DIR_GAN+'experimento_1'+'/'+SYMBOL
    # TGAN_PATH_1=TGAN_DIR_1+'/experiment_00/synthetic_data.h5'
    # gen1, real,pred, pred2,t_data = get_syntetic_data(symbol=SYMBOL,TYPE_INPUT=TYPE_INPUT_exp1,
    #                                                 OUTPUT_FEATURES=OUTPUT_FEATURES,gen_model=TGAN_PATH_1,DATA_DIR=DIR_DATA,
    #                                                 lstm_model_1=lstm_model,lstm_model_2=lstm_model2,is_pred=True,is_gen=False)
    # #experimento 2
    TYPE_INPUT_exp2_3='PREV_WINDOW'
    #TYPE_DATA_2_3 = 'Most_Important_Features/'
    TYPE_DATA_2_3='Complete/'
    DIR_DATA=DATA_PATH+TYPE_DATA_2_3
    #n_gan_epoch=600
    TGAN_DIR_2=SAVE_DIR_GAN+'experimento2_2'
    gen2,real2= get_syntetic_data_tgan_global(SYMBOL=SYMBOL,TYPE_INPUT=TYPE_INPUT_exp2_3,OUTPUT_TYPE=OUTPUT_TYPE,
                                                    OUTPUT_FEATURES=OUTPUT_FEATURES,DATA_DIR=DIR_DATA,gen_model_dir=TGAN_DIR_2)

    plot_CLOSE_TGAN_win(gen_full_df=gen2,len_win=10,num_win=5)
    plot_CLOSE_TGAN_win(gen_full_df=real2,len_win=10,num_win=5)
    #experimento 3

    TGAN_DIR_3=SAVE_DIR_GAN+'experimento2_3'
    gen3 ,real3= get_syntetic_data_tgan_global(SYMBOL=SYMBOL,TYPE_INPUT=TYPE_INPUT_exp2_3,OUTPUT_TYPE=OUTPUT_TYPE,
                                                    OUTPUT_FEATURES=OUTPUT_FEATURES,DATA_DIR=DIR_DATA,gen_model_dir=TGAN_DIR_3)
    plot_CLOSE_TGAN_win(gen_full_df=gen3,len_win=10,num_win=5)
    plot_CLOSE_TGAN_win(gen_full_df=real3,len_win=10,num_win=5)
    # # #Para comparar por ventana
    # # print('experimento 1')
    # # gen1_df,real_df=get_dataframe_for_compare(data=gen1,t_data=t_data,real_data=real,experiment=1,is_anchor=False,FEATURES=OUTPUT_FEATURES,is_plot=True)
    # #print('experimento 2')
    #
    # #gen2_df,real2_df=get_dataframe_for_compare(data=gen2,t_data=t_data2,real_data=real2,experiment=2,is_anchor=False,FEATURES=OUTPUT_FEATURES,is_plot=True)
    # #print('experimento 3')
    #
    # #gen3_df,real3_df=get_dataframe_for_compare(data=gen3,t_data=t_data3,real_data=real3,experiment=3,is_anchor=False,FEATURES=OUTPUT_FEATURES,is_plot=True)
    #
    #
    #pred_df,real_df=get_dataframe_for_compare(data=pred,data2=pred2,t_data=t_data,real_data=real,is_anchor=False,FEATURES=OUTPUT_FEATURES,is_plot=True)
    # #gen_df=get_dataframe(data=gen,t_data=t_data,is_anchor=False,FEATURES=OUTPUT_FEATURES)
    # # real_df=get_dataframe(data=real,t_data=t_data,is_anchor=False,FEATURES=OUTPUT_FEATURES)
    # #pred_df=get_dataframe(data=pred,t_data=t_data,is_anchor=False,FEATURES=OUTPUT_FEATURES)
    # # # plot_OHLC(gen_full_df=gen_df,real_full_df=real_df)
    # #plot_CLOSE(gen_full_df=gen_df,real_full_df=real_df)
    # # plot_OHLC(gen_full_df=pred_df,real_full_df=real_df)
    # #plot_CLOSE(gen_full_df=pred_df,real_full_df=real_df)
    # #plot_full_OHLC(generated_data=gen_df,real_data=real_df,predicted_data=pred_df)
    # #plot_full_CLOSE(gen_full_df=gen_df,real_full_df=real_df,pred_full_data=pred_df)
    #
    #
