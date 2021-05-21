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
from mae_count import count_mae


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict(): 
    df_residential = pd.read_csv('../Data/DF_Train.csv')

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


@app.route('/result', methods=['GET', 'POST'])
def result():
    try:
        print(request.method)
        if request.method == 'POST':
            input = request.form
            
            df_predict = pd.DataFrame({
                'BATHRM' : [int(input['bathrooms'])],
                'HF_BATHRM' : [int(input['half_bathrooms'])],
                'ROOMS' : [int(input['rooms'])],
                'BEDRM' : [int(input['bedrooms'])],
                'ayb_age' : [int(input['building_age'])],
                'eyb_age' : [int(input['renovation_years'])],
                'GBA' : [float(input['gba'])],
                'KITCHENS' : [int(input['kitchens'])],
                'FIREPLACES' : [int(input['fireplaces'])],
                'LANDAREA' : [float(input['landarea'])],
                'LATITUDE' : [float(input['latitude'])],
                'LONGITUDE' : [float(input['longitude'])],
                'AC' : [input['ac']],
                'QUALIFIED' : [input['qualified']],
                'WARD' : [input['ward']],
                'QUADRANT' : [input['quadrant']],
                'HEAT' : [input['heat']],
                'STYLE' : [input['style']],
                'USECODE' : [input['usecode']],
                'STRUCT' : [int(input['struct'])],
                'GRADE' : [int(input['grade'])],
                'CNDTN' : [int(input['condition'])],
                'ROOF' : [int(input['roof'])],
                'SALEYEAR' : [int(input['sale_year'])]
            })
            
    except:
        return predict_error()
    
    else:
        if request.method == 'POST':
            input = request.form
    
            df_predict = pd.DataFrame({
                'BATHRM' : [int(input['bathrooms'])],
                'HF_BATHRM' : [int(input['half_bathrooms'])],
                'ROOMS' : [int(input['rooms'])],
                'BEDRM' : [int(input['bedrooms'])],
                'ayb_age' : [int(input['building_age'])],
                'eyb_age' : [int(input['renovation_years'])],
                'GBA' : [float(input['gba'])],
                'KITCHENS' : [int(input['kitchens'])],
                'FIREPLACES' : [int(input['fireplaces'])],
                'LANDAREA' : [float(input['landarea'])],
                'LATITUDE' : [float(input['latitude'])],
                'LONGITUDE' : [float(input['longitude'])],
                'AC' : [input['ac']],
                'QUALIFIED' : [input['qualified']],
                'WARD' : [input['ward']],
                'QUADRANT' : [input['quadrant']],
                'HEAT' : [input['heat']],
                'STYLE' : [input['style']],
                'USECODE' : [input['usecode']],
                'STRUCT' : [int(input['struct'])],
                'GRADE' : [int(input['grade'])],
                'CNDTN' : [int(input['condition'])],
                'ROOF' : [int(input['roof'])],
                'SALEYEAR' : [int(input['sale_year'])]
            })
            
        prediction = int(model.predict(df_predict)[0])

        mae = count_mae()
    
        return render_template('result.html', spec=input, pred_result=prediction, mae=mae)


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
    model_filename = '../Model/Final_Model_RF.sav'
    model = pickle.load(open(model_filename, 'rb'))

    app.run(debug=True)

    

