import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt




def plot_full_OHLC(generated_data,real_data,predicted_data):
    gen_full_df = generated_data
    pred_full_df = predicted_data
    real_full_df = real_data
    fig = mpf.figure(style='yahoo', figsize=(14, 16))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    # av1 = fig.add_subplot(2, 3, 4, sharex=ax1)
    # av2 = fig.add_subplot(2, 3, 5, sharex=ax2)
    # av3 = fig.add_subplot(2, 3, 6, sharex=ax3)

    mpf.plot(gen_full_df, type='candle', ax=ax1, mav=(10, 20), axtitle='generated data')
    mpf.plot(real_full_df, type='candle', ax=ax2, mav=(10, 20), axtitle='real data')
    mpf.plot(pred_full_df, type='candle', ax=ax3, mav=(10, 20), axtitle='predicted data')

    mpf.show()

def plot_full_CLOSE(gen_full_df,real_full_df,pred_full_data):
    dataFrame = pd.concat([gen_full_df['close'], real_full_df['close'],pred_full_data['close']], axis=1)

    #dataFrame.plot(kind='scatter', x='Tiempo', y='US-Dollar', color='red')
    dataFrame.plot()
    plt.show()

def plot_OHLC(gen_full_df,real_full_df):

    fig = mpf.figure(style='yahoo', figsize=(14, 16))
    ax1 = fig.add_subplot(2, 1,1)
    ax2 = fig.add_subplot(2, 1,2)

    mpf.plot(gen_full_df,type='candle', ax=ax1, mav=(10, 20), axtitle='generated data')
    mpf.plot(real_full_df,type='candle', ax=ax2, mav=(10, 20), axtitle='real data')
    mpf.show()

def plot_CLOSE_lstm(gen_full_df,gen_full_df2,real_full_df):
    gen_full=pd.DataFrame()
    gen_full['pred_Tipo_1']=gen_full_df['Close']
    gen_full['pred_Tipo_2'] = gen_full_df2['Close']
    gen_full['real_close']=real_full_df['Close']
    dataFrame=gen_full.copy()
    #dataFrame = pd.concat([gen_full, real_full_df['real_close']], axis=1)
    dataFrame['Tiempo']=dataFrame.index
    ax = plt.gca()
    dataFrame.plot(kind='line', x='Tiempo', y='pred_Tipo_1', color='blue', ax=ax)
    dataFrame.plot(kind='line', x='Tiempo', y='pred_Tipo_2',color='green', ax=ax)
    dataFrame.plot(kind='line', x='Tiempo', y='real_close', color='red', ax=ax)
    #dataFrame.plot()
    plt.show()


def plot_CLOSE_TGAN(gen_full_df,real_full_df,experiment):
    gen_full=pd.DataFrame()
    gen_full['TGAN_{}'.format(experiment)]=gen_full_df['close']
    gen_full['real_close']=real_full_df['close']
    dataFrame=gen_full.copy()
    #dataFrame = pd.concat([gen_full, real_full_df['real_close']], axis=1)
    dataFrame['Tiempo']=dataFrame.index
    ax = plt.gca()
    dataFrame.plot(kind='line', x='Tiempo', y='TGAN_{}'.format(experiment),color='green', ax=ax)
    dataFrame.plot(kind='line', x='Tiempo', y='real_close', color='red', ax=ax)
    #dataFrame.plot()
    plt.show()

def plot_CLOSE_TGAN_win(gen_full_df,len_win,num_win):
    time=gen_full_df.index
    print(time)
    total_win=int(len(time)/len_win)
    pivot = 0
    n_win=0
    for i in range(total_win):
        if n_win>num_win:
            break
        print(time[pivot:pivot+len_win])
        win_df=gen_full_df.loc[time[pivot:pivot+len_win]]
        print(win_df)
        win_df.plot()
        plt.show()
        pivot+=len_win
        n_win+=1

