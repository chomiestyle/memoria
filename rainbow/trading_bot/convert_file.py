import tensorflow as tf
from tensorflow.keras.models import save_model, Sequential

model_path = r"/rainbow/models/AAPL_1D_tdqn_1"

model = tf.keras.models.load_model(model_path)

#save_model(model,model_path + r"\new_model.h5", overwrite=True,save_format='h5')