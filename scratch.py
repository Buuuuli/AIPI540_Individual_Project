from Call_Function import *
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from Call_Function import *
import pickle as pkl
import torch

# ML Part
#filepath = 'Model/randomforest.sav'

#loaded_model = pkl.load(open(filepath, 'rb'))

#age_input = 55
#sex_input = 'male'
#localization_input = 'chest'


#a = meta_pipeline(age_input,sex_input,localization_input)


#test_preds = loaded_model.predict(a).reshape(len(a),1)


#meta__prob = loaded_model.predict_proba(a)



#if test_preds[0][0] ==4:
    #print('Including Melanoma')
#else:
    #print('Benign Keratosis')





#DL Part



model2 = torch.load('Model/deep_fullmodel_new.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = 'webelement/ISIC_0024315.jpg'


b = image_pipeline(image)


j,k = test_model(model2, b, device)





import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
#import dash_table_experiments as dt
import base64
import datetime
import json
import pandas as pd
import plotly
import io
import numpy as np




app = dash.Dash()

app.scripts.config.serve_locally = True

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])


def parse_contents(contents):
    return html.Div([

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[:100] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')])
def update_output(images):
    if not images:
        return

    for i, image_str in enumerate(images):
        image = image_str.split(',')[1]
        data = base64.decodestring(image.encode('ascii'))
        with open(f"image_{i+1}.jpg", "wb") as f:
            f.write(data)

    children = [parse_contents(i) for i in images]
    return children


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)

