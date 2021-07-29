import pandas
import matplotlib.pyplot as plt
import pandas as pd

name_stocks=['AAPL','CSCO','CVX','IBM','INTC']
saved_dir='C:/Users/eduar/PycharmProjects/Comparacion_memoria/Forecasting/LSTM/restults/'
for j in name_stocks:
    result=pd.read_csv(saved_dir+'results_{}.csv'.format(j))
    result=result[['{}_predict'.format(j),'{}_actual'.format(j)]]
    result.plot()
    plt.show()



# # Initialise the subplot function using number of rows and columns
# figure, axis = plt.subplots(2, 2)
#
# # For Sine Function
# axis[0, 0].plot(X, Y1)
# axis[0, 0].set_title("Sine Function")
#
# # For Cosine Function
# axis[0, 1].plot(X, Y2)
# axis[0, 1].set_title("Cosine Function")
#
# # For Tangent Function
# axis[1, 0].plot(X, Y3)
# axis[1, 0].set_title("Tangent Function")
#
# # For Tanh Function
# axis[1, 1].plot(X, Y4)
# axis[1, 1].set_title("Tanh Function")
#
# # Combine all the operations and display
# plt.show()



