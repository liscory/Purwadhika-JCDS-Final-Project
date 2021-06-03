from flask import Flask, render_template, request
import requests
import joblib
import pickle
import plotly
import plotly.graph_objs as go
import json
import pandas as pd 
import numpy as np
from model_plots import show_map, show_bar_ward, show_bar_ward_count, show_bar_grade, show_bar_grade_count, show_bar_usecode, show_bar_usecode_count, show_line_sale
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error


class NoTransformer(BaseEstimator):
    """Passes through data without any change and is compatible with ColumnTransformer class"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X
    

with open('../Model/Final_Model_Catboost_v2.sav' ,'rb') as f:
    model_dict = pickle.load(f)
    

def count_mae():
    df_train = pd.read_csv('../Data/DF_Train_Clean.csv')
    X_train = df_train.drop('PRICE', axis=1)
    y_train = df_train['PRICE']
    y_pred = model_dict['model'].predict(model_dict['transformer'].transform(X_train))
    mae = mean_absolute_error(y_train, y_pred)
    return mae

MAE = count_mae()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict(): 
    df_residential = pd.read_csv('../Data/DF_Train_Clean.csv')

    ward = sorted(df_residential['WARD'].unique().tolist())
    quadrant = df_residential['QUADRANT'].unique().tolist()
    heat = df_residential['HEAT'].unique().tolist()
    style = sorted(df_residential['STYLE'].unique().tolist())
    usecode = sorted(df_residential['USECODE'].unique().tolist())
    saleyear = sorted(df_residential['SALEYEAR'].unique().tolist(), reverse=True)

    house_spec = {
        'ward':ward,
        'quadrant':quadrant,
        'heat':heat,
        'style':style,
        'usecode':usecode,
        'saleyear':saleyear
        }

    return render_template('predict.html', spec=house_spec)


@app.route('/result', methods=['POST'])
def result():
    try:
        print(request.method)
        if request.method == 'POST':
            form_input = request.form
            
            df_predict = pd.DataFrame({
                #Int value but changed into float to avoid error if the user inputs the number in float format (e.g., 1.0)
                'BATHRM' : [float(form_input['bathrooms'])], # Int
                'HF_BATHRM' : [float(form_input['half_bathrooms'])], # Int
                'ROOMS' : [float(form_input['rooms'])], # Int
                'BEDRM' : [float(form_input['bedrooms'])], # Int
                'ayb_age' : [float(form_input['building_age'])], # Int
                'eyb_age' : [float(form_input['renovation_years'])], # Int
                'GBA' : [float(form_input['gba'])], # Int
                'KITCHENS' : [float(form_input['kitchens'])], # Int
                'FIREPLACES' : [float(form_input['fireplaces'])], # Int
                'LANDAREA' : [float(form_input['landarea'])], # Int
                'LATITUDE' : [float(form_input['latitude'])],
                'LONGITUDE' : [float(form_input['longitude'])],
                'AC' : [form_input['ac']],
                'QUALIFIED' : [form_input['qualified']],
                'WARD' : [form_input['ward']],
                'QUADRANT' : [form_input['quadrant']],
                'HEAT' : [form_input['heat']],
                'STYLE' : [form_input['style']],
                'USECODE' : [form_input['usecode']],
                'STRUCT' : [int(form_input['struct'])],
                'GRADE' : [int(form_input['grade'])],
                'CNDTN' : [int(form_input['condition'])],
                'ROOF' : [int(form_input['roof'])],
                'SALEYEAR' : [int(form_input['sale_year'])]
            })

            prediction = model_dict['model'].predict(model_dict['transformer'].transform(df_predict))[0]
        
            return render_template('result.html', spec=form_input, pred_result=prediction, mae=MAE)
  
    except:
        return predict_error()


@app.route('/data', methods=['GET', 'POST'])
def data():
    datamap = show_map()
    dataline_sale = show_line_sale()
    databar_ward = show_bar_ward()
    databar_ward_count = show_bar_ward_count()
    databar_grade = show_bar_grade() 
    databar_grade_count = show_bar_grade_count()
    databar_usecode = show_bar_usecode()
    databar_usecode_count = show_bar_usecode_count()
    
    return render_template('data.html', 
                           datamap = datamap,
                           dataline_sale=dataline_sale,
                           databar_ward=databar_ward, 
                           databar_ward_count=databar_ward_count,
                           databar_grade = databar_grade, 
                           databar_grade_count = databar_grade_count,
                           databar_usecode = databar_usecode,
                           databar_usecode_count = databar_usecode_count)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict-error', methods=['GET', 'POST'])
def predict_error():
    return render_template('predict-error.html')


if __name__ == '__main__':
    app.run(debug=True)