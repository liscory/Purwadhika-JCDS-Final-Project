import pickle
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_error

model_filename = '../Model/Final_Model_RF.sav'
model = pickle.load(open(model_filename, 'rb'))

def count_mae():
    df_train = pd.read_csv('../Data/DF_train.csv')
    X_train = df_train.drop('PRICE', axis=1)
    y_train = df_train['PRICE']
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    return mae