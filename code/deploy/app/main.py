
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash_table import DataTable
from dash_canvas import DashCanvas
from dash_canvas.utils import (array_to_data_url, parse_jsonstring,
                              watershed_segmentation)
from skimage import io, color, img_as_ubyte
import numpy as np
import dash_core_components as dcc
import time
import json
import requests
from PIL import Image
import urllib
import pandas as pd
#!pip install jupyter-plotly-dash pip install dash_daq dash_canvas jupyterlab-dash
#!jupyter labextension install jupyterlab-dash
#import jupyterlab_dash
#viewer = jupyterlab_dash.AppViewer()

app = dash.Dash(__name__)
# app.config.suppress_callback_exceptions = True

rocf_url = 'https://github.com/Cerebrock/ROCF/raw/master/imgs/ROCF.png'
####
bg_url = 'https://giraict.rw/wp-content/uploads/2018/01/White-VBA1-600x600.jpg'

#img = io.imread(filename, as_gray=True)

colors = {'background': 'rgba(7,74,116,0.05)',
          'text': 'rgba(70, 135, 246, 1)',
          'paper_bgcolor':'rgba(0,0,0,0)',
          'plot_bgcolor':'rgba(0,0,0,0)'}
          #rgba(7, 74, 150,1)
          #rgba(70, 135, 246, 1)
          #rgba(37, 102, 143, 0.2)

def get_img(filename, height=None, width=None):
    r = requests.get(filename, stream=True)
    img = Image.open(r.raw)
    if height != None: 
        img = img.resize((width, height))
    img = np.array(img.convert('L'))
    height, width = img.shape
    return img, height, width 

img, height_rocf, width_rocf = get_img(rocf_url)
canvas_width = 800
canvas_height = 800
scale = canvas_width / width_rocf

times = []
columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path', 'time']

app.layout = html.Div(style={'backgroundColor': 
                            colors['background'], 
                            'text-align': 'center'},
                      children=[
                      html.Div(id='canvas_div', 
                               children=[
                                   html.H1('Observa atentamente la figura'),
                                   html.P("Cuando la imagen desaparezca, copiala lo mejor posible dibujando sobre la pantalla"),  
                                   DashCanvas(
                                        id='canvas',
                                        lineColor='#00243C',
                                        width=canvas_width,
                                        lineWidth=5,
                                        height=canvas_height,
                                        image_content=array_to_data_url(img),
                                        #scale=1,
                                        hide_buttons=['line', 'zoom', 'pan', 'rectangle', 'pencil', 'select', 'redo', 'undo'],
                                        goButtonTitle='Finalizar'),
                                   html.A(html.Button('Descargar datos', id='download-button'),
                                                    id='download-a',
                                                    download='download.csv',
                                                    #href="",
                                                    n_clicks=0,
                                                    #target='_blank'
                                                    )
                                    ],
                               style={'text-align':"center",
                                      'border': '2px solid black'}),
                      html.Div(id='table_div',
                               style={'display':'none'},
                               children=[DataTable(id='canvas-table',
                                        style_cell={'text-align': 'left'},
                                        columns=[{"name": i, "id": i} for i in columns]),
                                        ]),
                      dcc.Interval(id='interval', max_intervals=1, interval=1000),
                      dcc.Store(id='memory', 
                                storage_type='memory')
                      ])

# https://dash.plot.ly/canvas

#@app.callback(Output('canvas-color', 'lineColor'),
#            [Input('color-picker', 'value')])
#def update_canvas_linewidth(value):
#    if isinstance(value, dict):
#        return value['hex']
#    else:
#        return value


#@app.callback(Output('canvas-color', 'lineWidth'),
#            [Input('bg-width-slider', 'value')])
#def update_canvas_linewidth(value):
#    return value


@app.callback([Output('download-a', 'href'),
               Output('download-a', 'download')],
              [Input('download-a', 'n_clicks')],
              [State('memory', 'data')])
def download(inp, data):
    if (inp is None) or (inp%3==0):
        raise PreventUpdate
    df = pd.DataFrame(data)
    csv_string = df.to_csv(encoding='utf-8', float_format='%.4g')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return csv_string, f"{time.strftime('%m/%d_%H:%M:%S')}.csv"


@app.callback([Output('memory', 'data'), 
               Output('canvas-table', 'data')],
              [Input('canvas', 'json_data')])
def update_data(string):
    if string:
        data = json.loads(string)
        times.append(time.time())
        data['objects'] = [{k:v for k,v in list(l.items()) + [('time', t)]} for l,t in zip(data['objects'], times)]
    else:
        raise PreventUpdate
    
    return {'data': data['objects']}, data['objects']

@app.callback([Output('canvas', 'image_content'),
               Output('canvas', 'json_data')],
              [Input('interval', 'n_intervals')])
def update_(n):
    time.sleep(5)
    print('Image updated\n')
    img, height, width = get_img(bg_url, height_rocf, width_rocf)
    return (array_to_data_url(img), '')

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', 
                   debug=True,
                   port='80')
