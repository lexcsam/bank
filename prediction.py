from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.metrics import Precision, Recall
from keras import backend as K
import pickle
import pandas as pd
import numpy as np



def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Load the model from the .h5 file
loaded_model = load_model('updated_model.h5',custom_objects={'f1_m': f1_m})

with open('scaler_object.pkl', 'rb') as f:
    scaler_fit_data = pickle.load(f)





async def prediction(_dict):
    new_df=pd.DataFrame([_dict])
    scaledData = scaler_fit_data.transform(new_df)
    model_predict=loaded_model.predict(scaledData)
    class_label = model_predict.argmax(axis=-1)
    print(class_label)
    return class_label
