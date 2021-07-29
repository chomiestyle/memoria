from Forecasting.TimeGAN.TGAN_future import *

DATA_PATH='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Data/combined/'

#TYPE_DATA='Most_Important_Features/'
#TYPE_DATA='Full_data/'
#TYPE_DATA='data_assets_TI/'
#TYPE_DATA='OHLC_NEWS/'
#TYPE_DATA_ARRAY=['Full_data/','Most_Important_Features/','data_assets_TI/','OHLC_NEWS/']
#TYPE_DATA_ARRAY=['Most_Important_Features/']

SAVE_DIR='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/TimeGAN/TRAIN/'
OUTPUT_TYPE='CLOSE'
for SYMBOL in ['INTC','MSFT']:
#for SYMBOL in ['IBM']:
    #EXPERIMENT 1
    # SAVE_DIR_1=SAVE_DIR+'experimento_1'+'/'+'{}'.format(SYMBOL)
    # TYPE_DATA='data_assets_TI/'
    # TYPE_INPUT = 'RANDOM'
    # model_exp_1 = TGAN(experiment=1, save_model=True, seq_len=5,save_dir=SAVE_DIR_1)
    # model_exp_1.training_loop(SYMBOL=SYMBOL, TYPE_INPUT=TYPE_INPUT, OUTPUT_FEATURES=['close'],
    #                                DIR_DATA=DATA_PATH + TYPE_DATA, is_separated=False, plot_data=False)


    #EXPERIMENTO 2 Y 3
    #TYPE_DATA_ARRAY=['Most_Important_Features/']
    TYPE_DATA_ARRAY = ['Complete/']
    #training_steps = [ 1000, 1200, 1500,1800, 2000]
    training_steps = [ 600, 900]
    for TYPE_DATA in TYPE_DATA_ARRAY:

        TYPE_INPUT = 'PREV_WINDOW'
        for i in training_steps:
            SAVE_DIR_2 = SAVE_DIR + 'experimento2_2' + '/' + SYMBOL + '/' + 'TimeGAN_{}_{}_{}'.format(TYPE_INPUT,
                                                                                                    OUTPUT_TYPE, i)
            model_exp_2 = TGAN(experiment=2,train_step=i,save_model=True,seq_len=5,save_dir=SAVE_DIR_2)
            model_exp_2.training_loop(SYMBOL=SYMBOL,TYPE_INPUT=TYPE_INPUT,OUTPUT_FEATURES=['close'],
                                  DIR_DATA=DATA_PATH+TYPE_DATA,is_separated=False, plot_data=False)

            SAVE_DIR_3= SAVE_DIR + 'experimento2_3' + '/' + SYMBOL + '/' + 'TimeGAN_{}_{}_{}'.format(TYPE_INPUT,
                                                                                                    OUTPUT_TYPE, i)
            model_exp_3 = TGAN(experiment=3,train_step=i,save_model=True,seq_len=5,save_dir=SAVE_DIR_3)
            model_exp_3.training_loop(SYMBOL=SYMBOL,TYPE_INPUT=TYPE_INPUT,OUTPUT_FEATURES=['close'],
                                  DIR_DATA=DATA_PATH+TYPE_DATA,is_separated=False, plot_data=False)