import pandas as pd

from rainbow.eval import main
from Forecasting.get_syntetic_data import *
from rainbow.trading_bot.agent import Agent
from rainbow.trading_bot.methods import train_model, evaluate_model

def render(history_1,Tipo):
    color='orange'
    prices=[]
    actions=[]
    for step in history_1:
        #print(step)
        price=step[0]
        action=step[1]
        #print(price)
        #print(action)
        prices.append(price)
        actions.append(action)
    p=pd.DataFrame()
    p['price']=prices
    p['actions']=actions
    #print(p)
    buy=p[p['actions']=='BUY']
    sell=p[p['actions']=='SELL']

    #fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    #fig.suptitle("Performance")

    plt.plot(np.arange(len(p)), p['price'].values, label="price", color=color)
    plt.scatter(buy.index, buy['price'], marker="^", color="green")
    plt.scatter(sell.index, sell['price'], marker="^", color="red")
    plt.title("Experimento de Trading para {}".format(Tipo))
    plt.xlabel("Dias tasado")
    plt.ylabel("Price (USD)")
    plt.show()

def trading_experiment(SYMBOL='IBM'):
    manual_run = False
    DATA_PATH = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Data/combined/'

    # #TYPE_DATA='Most_Important_Features/'
    # #TYPE_DATA='Full_data'
    # #TYPE_DATA='data_assets_TI'
    TYPE_DATA = 'OHLC_NEWS/'

    #TYPE_DATA_LSTM = 'data_assets_TI/'
    TYPE_DATA_LSTM = 'Complete/'
    OUTPUT_TYPE = 'CLOSE'
    SAVE_DIR_LSTM = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/TRAIN/'
    LSTM_DIR = SAVE_DIR_LSTM + TYPE_DATA_LSTM + '/' + SYMBOL + '/' + 'lstm_{}.h5'.format(OUTPUT_TYPE)
    lstm_model = LSTM_DIR
    LSTM_DIR2 = SAVE_DIR_LSTM + TYPE_DATA + '/' + SYMBOL + '/' + 'lstm_{}.h5'.format(OUTPUT_TYPE)
    lstm_model2 = LSTM_DIR2
    OUTPUT_FEATURES = ['Close']
    # #GAN
    SAVE_DIR_GAN = 'C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/TimeGAN/TRAIN/'
    #
    # experimento 1
    # TYPE_INPUT_exp1='RANDOM'
    TYPE_INPUT_exp1 = 'PREV_WINDOW'
    #TYPE_DATA_1 = 'data_assets_TI/'
    DIR_DATA = DATA_PATH + TYPE_DATA
    TGAN_DIR_1 = SAVE_DIR_GAN + 'experimento_1' + '/' + SYMBOL
    TGAN_PATH_1 = TGAN_DIR_1 + '/experiment_00/synthetic_data.h5'
    gen1, real, pred, pred2, t_data = get_syntetic_data(symbol=SYMBOL, TYPE_INPUT=TYPE_INPUT_exp1,
                                                        OUTPUT_FEATURES=OUTPUT_FEATURES, gen_model=TGAN_PATH_1,
                                                        DATA_DIR=DIR_DATA,
                                                        lstm_model_1=lstm_model, lstm_model_2=lstm_model2, is_pred=True,
                                                        is_gen=False)

    # print('real')
    # print(real)
    # print('pred')
    # print(pred2)
    model_name = 'double-dqn_1D_with_prediction'
    #model_name='C:/Users/eduar/PycharmProjects/Comparacion_memoria/rainbow/models/double-dqn_1D_with_prediction'
    pretrained=True
    window_size=5
    debug = False
    strategy = 'double-dqn'
    agent = Agent(2 * window_size - 1, strategy=strategy, pretrained=pretrained, model_name=model_name, manual=manual_run)

    #Aca comeinzan los experimentos
    #Los con LSTM
    pred_val_profit, pred_val_history= evaluate_model(agent, pred, window_size, debug)
    print('prediction profit')
    print(pred_val_profit)
    print('prediction history')
    print(pred_val_history)
    pred2_val_profit, pred2_val_history= evaluate_model(agent, pred2, window_size, debug)
    print('prediction 2 profit')
    print(pred2_val_profit)
    print('prediction  2history')
    print(pred2_val_history)
    real_val_profit, real_val_history= evaluate_model(agent, real, window_size, debug)
    print('real profit')
    print(real_val_profit)
    print('real history')
    print(real_val_history)
    render(real_val_history,Tipo='Valores Reales')
    render(pred_val_history,Tipo='Predicciones LSTM Tipo 1')
    render(pred2_val_history,Tipo='Predicciones LSTM Tipo 2')
    #gen_profit, gen_history ,agent_1 = main(pred, 5, "AAPL_1D_tdqn_1", True, manual_run, per_value='day',is_prediction=True)
    #print(gen_history)
    # print(agent_1.inventory)
    # real_profit, real_history ,agent_2= main(real, 5, "AAPL_1D_tdqn_1", True, manual_run, per_value='day',is_prediction=False)
    # print(real_history)
    # print(agent_2.inventory)
    # print('ganancias con el agente y valores reales')
    # print(real_profit)
    # print('ganancias con el agente y valores generados')
    # print(gen_profit)

trading_experiment(SYMBOL='WMT')