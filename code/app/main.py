
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

#!pip instal jupyter-plotly-dash pip install dash_daq dash_canvas jupyterlab-dash
#!jupyter labextension install jupyterlab-dash

import json

#import jupyterlab_dash
#viewer = jupyterlab_dash.AppViewer()

app = dash.Dash(__name__)
# app.config.suppress_callback_exceptions = True

filename = 'https://github.com/Cerebrock/ROCF/raw/master/imgs/ROCF.png'
canvas_width = 800
####

img = io.imread(filename, as_gray=True)

columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']

app.layout = html.Div([
        html.Div([
            DashCanvas(
                id='canvas',
                width=canvas_width,
                filename=filename,
                hide_buttons=['line', 'zoom', 'pan', 'rectangle', 'pencil', 'select'],
                ),

            DataTable(id='annot-canvas-table',
                style_cell={'textAlign': 'left'},
                columns=[{"name": i, "id": i} for i in columns]),
                ], className="six columns"),
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


@app.callback(Output('annot-canvas-table', 'data'),
              [Input('canvas', 'json_data')])
def update_data(string):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate
    return data['objects'][1:]


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', 
                   debug=False,
                   port='80')
