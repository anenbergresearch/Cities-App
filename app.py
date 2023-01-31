import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dah import Dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
df = pd.read_csv('updated_unified_data.csv')
c40 = pd.read_excel('cityid-c40_crosswalk.xlsx')
ID_pop = df[df['Year']==2019][['ID','Population']]
c40_p = c40.merge(ID_pop, how = 'left', left_on = 'city_id', right_on ='ID')
total = c40_p[['ID','c40','continent']].merge(df, how = 'right', on ='ID')
df = total[['ID','City','c40','Country','continent','Year','Population','NO2','PM','O3']].copy()
df['CityCountry'] = df.City + ', ' + df.Country + ' (' +df.ID.apply(int).apply(str) +')'

import dash.dependencies


pd.options.plotting.backend = "plotly"



app = Dash(__name__)#, external_stylesheets=external_stylesheets)


available_indicators = ['O3','PM','NO2']

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='NO2'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': 'Population', 'value': 'Population'}],
                value='Population'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],                
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),
    

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'hovertext': 'Tokyo, Japan (13017)'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        id='crossfilter-year--slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()},
        step=None
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),
    html.Div(dcc.RangeSlider(
        id='pop-limit--slider',
        min=df['Population'].min(),
        max=df['Population'].max(),
        value=[df['Population'].min(),df['Population'].max()],
        allowCross=False,
        marks=None,
        tooltip={"placement": "bottom","always_visible": True}

    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),
])

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),

    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-year--slider', 'value'),
     dash.dependencies.Input('pop-limit--slider', 'value'),
     ])


def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value,pop_limit):
    dff = df.query('Year == @year_value')
    dff = dff.query('@pop_limit[0] < Population <@pop_limit[1]')
    
    fig = px.scatter(dff, x=xaxis_column_name,
            y=yaxis_column_name,
            hover_name='CityCountry',
            color = 'continent', symbol='c40'
            )
    j=0
    k=0
    for i, trace in enumerate(fig.data):
        name = trace.name.split(',')
        if name[1] == ' not_c40':
            trace['name'] = name[0]
            trace['showlegend']=True
            trace['legendgroup']='Not C40'
            if j==0:
                trace['legendgrouptitle_text']='Not C40 Cities'
                j+=1
        else:
            trace['name'] = name[0]
            trace['legendgroup']='C40'
            if k==0:
                trace['legendgrouptitle_text']='C40 Cities'
                k+=1
    fig.update_layout(legend=dict(groupclick="toggleitem"))

        
    fig.update_layout(legend_title_text='')


    fig.update_traces(customdata=dff['CityCountry'])
    
    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title, axiscol_name):

    fig = px.scatter(dff, x='Year', y=axiscol_name)

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig
@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    dff = df[df['CityCountry'] == hoverData['points'][0]['hovertext']]
    country_name = dff['CityCountry'].iloc[0]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title, xaxis_column_name)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['CityCountry'] == hoverData['points'][0]['hovertext']]
    country_name = dff['CityCountry'].iloc[0]
    return create_time_series(dff, axis_type, country_name,yaxis_column_name)

if __name__== '__main__':
    app.run_server(debug=True)