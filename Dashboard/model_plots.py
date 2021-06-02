import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
import plotly.graph_objs as go
import json
import folium
from folium.features import DivIcon


df = pd.read_csv('../Data/DF_Residential_Analysis.csv')


def show_map():
    df_ward = df[['WARD', 'PRICE']].groupby(['WARD']).median().reset_index().rename(columns={'WARD':'Ward',
                                                                                             'PRICE':'Median Price'})
    
    with open('../Data/ward-2012.geojson') as f:
        geojson_data = json.load(f)
    
    fig = go.Figure(data=go.Choropleth(geojson=geojson_data,
                                        colorscale='BuPu',
                                        locations=df_ward['Ward'].astype(str),
                                        featureidkey = "properties.NAME",
                                        z = df_ward['Median Price'].astype(float))) #To color-code data
    
    fig.update_geos(fitbounds="locations", visible=False)
    
    fig.update_layout(title_text = 'Washington DC Ward Map',
                      geo_scope='usa', # Limit map scope to USA
                      font_family="Open Sans",
                      title_font_family="Open Sans"
                      )
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json
    

def show_bar_ward():
    # Version 1 
    
    df_ward = df[['WARD', 'PRICE']].groupby(['WARD']).median().reset_index().rename(columns={'WARD':'Ward',
                                                                                            'PRICE':'Median Price'})
    
    fig = px.bar(df_ward,
                 x='Ward',
                 y='Median Price',
                 barmode='group',
                 color_discrete_sequence=["darkslateblue"],
                 title='Median Property Price by Ward'
                 )
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'title_font_family':'Open Sans'
                       })
    
    # Version 2
    '''
    df_ward = df[['WARD', 'PRICE']].groupby(['WARD']).median().reset_index()
    fig = go.Figure(data=go.Bar(x=df_ward['WARD'], 
                                    y=df_ward['PRICE'], 
                                    marker=dict(color='darkslateblue')
                                    ))
    
    fig.update_layout(title ='Median Property Price by Ward', 
                      xaxis = {'showgrid': False,
                                'title':'Ward'},
                      yaxis = {'showgrid': False,
                               'title':'Median Price'},
                      plot_bgcolor = 'rgba(0, 0, 0, 0)',
                      paper_bgcolor = 'rgba(0, 0, 0, 0)'
                      )
    '''
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def show_line_sale():
    df_sale_year = df[['SALEYEAR', 'PRICE']].groupby(['SALEYEAR']).median().reset_index()
    
    fig = go.Figure(data=go.Scatter(x=df_sale_year['SALEYEAR'], 
                                    y=df_sale_year['PRICE'], 
                                    mode='lines', # To connect the points
                                    marker=dict(color='darkslateblue')
                                    ))
    
    fig.update_layout(title ='Median Property Price Each Year', 
                      xaxis = {'showgrid': False,
                                'title':'Sale Year'},
                      yaxis = {'showgrid': False,
                               'title':'Median Price'},
                      plot_bgcolor = 'rgba(0, 0, 0, 0)',
                      paper_bgcolor = 'rgba(0, 0, 0, 0)'
                      )
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def show_bar_ward_count():
    
    df_ward = df[['WARD', 'PRICE']].groupby(['WARD']).count().reset_index().rename(columns={'WARD':'Ward',
                                                                                            'PRICE':'Count'})
    
    fig = px.bar(df_ward,
                 x='Ward',
                 y='Count',
                 barmode='group',
                 color_discrete_sequence=["darkslateblue"],
                 title='Number of Properties by Ward'
                 )
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'title_font_family':'Open Sans'
                       })
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def show_bar_grade():
    # Version 1 
    
    df_grade = df[['GRADE', 'PRICE']].groupby(['GRADE']).median().reset_index().rename(columns={'GRADE':'Grade',
                                                                                               'PRICE':'Median Price'})
    
    fig = px.bar(df_grade,
                 x='Grade',
                 y='Median Price',
                 barmode='group',
                 color_discrete_sequence=["darkslateblue"],
                 title='Median Property Price by Grade',
                                  category_orders={'Grade':['Fair Quality',
                                           'Average',
                                           'Above Average',
                                           'Good Quality',
                                           'Very Good',
                                           'Excellent',
                                           'Superior']}
                 )
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'title_font_family':'Open Sans'
                       })
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def show_bar_grade_count():
    
    df_grade = df[['GRADE', 'PRICE']].groupby(['GRADE']).count().reset_index().rename(columns={'GRADE':'Grade',
                                                                                              'PRICE':'Count'})
    
    fig = px.bar(df_grade,
                 x='Grade',
                 y='Count',
                 barmode='group',
                 color_discrete_sequence=["darkslateblue"],
                 title='Number of Properties by Grade',
                 category_orders={'Grade':['Fair Quality',
                                           'Average',
                                           'Above Average',
                                           'Good Quality',
                                           'Very Good',
                                           'Excellent',
                                           'Superior']}
                 )
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'title_font_family':'Open Sans'
                       })
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def show_bar_usecode():
    
    df_usecode = df[['USECODE', 'PRICE']].groupby(['USECODE']).median().reset_index().rename(columns={'USECODE':'Use Code',
                                                                                                    'PRICE':'Median Price'})
    df_usecode['Use Code'] = df_usecode['Use Code'].astype('str')
    
    fig = px.bar(df_usecode,
                 x='Use Code',
                 y='Median Price',
                 barmode='group',
                 color_discrete_sequence=["darkslateblue"],
                 title='Median Property Price by Use Code'
                 )
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'title_font_family':'Open Sans'
                       })
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def show_bar_usecode_count():
    
    df_usecode = df[['USECODE', 'PRICE']].groupby(['USECODE']).count().reset_index().rename(columns={'USECODE':'Use Code',
                                                                                                    'PRICE':'Count'})
    df_usecode['Use Code'] = df_usecode['Use Code'].astype('str')
    
    fig = px.bar(df_usecode,
                 x='Use Code',
                 y='Count',
                 barmode='group',
                 color_discrete_sequence=["darkslateblue"],
                 title='Number of Properties by Use Code'
                 )
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'title_font_family':'Open Sans'
                       })
    
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json
