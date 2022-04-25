import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from Call_Function import *
import pickle as pkl
import torch
import base64
import io
from PIL import Image

app = dash.Dash(__name__)
server = app.server
doc_image = "webelement/docnew.png"
test_base64 = base64.b64encode(open(doc_image, 'rb').read()).decode('ascii')

app.layout = html.Div([
    html.Div([html.H1('Skin Health')], style={'color': 'blue', 'fontSize': 14, 'textAlign': 'center',
                                              'marginBottom': 50, 'marginTop': 25}),
    html.Div([

    html.Img(src='data:image/png;base64,{}'.format(test_base64))], style={'textAlign': 'center'}),
    html.Div([
        html.H4("Step 1 Choose Your Gender",style={'textAlign': 'center'
                                              }),
        html.Div(
            dcc.Dropdown(
                ["female", "male", "unknown"],
                "male",
                id='gender', style={'textAlign': 'center',
                                              'marginBottom': 50, 'marginTop': 25}
            ), style={
                       'position': 'absolute',
                             'left' : '50%',
                             'transform': 'translateX(-50%)',
                             'display': 'inline-block',
                              'width': '50%',
                             }),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H4("Step 2 Where is the symptom",style={'textAlign': 'center'
                                              }),
        html.Br(),
        html.Div(
            dcc.Dropdown(
                ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
                 'genital', 'hand', 'lower extremity', 'neck', 'scalp',
                 'trunk', 'unknown', 'upper extremity'],
                "chest",
                id='localization',
                style={'textAlign': 'center'}

            ), style={
                       'position': 'absolute',
                             'left' : '50%',
                             'transform': 'translateX(-50%)',
                             'display': 'inline-block',
                              'width': '50%',
                             }),
        html.Br(),
        html.Br(),
        html.H4("Step 3 Enter your age",style={'textAlign': 'center',
                                              }),
        dcc.Input(id='age', value=30, type='number', style={'textAlign': 'center',
                       'position': 'absolute',
                             'left' : '50%',
                             'transform': 'translateX(-50%)',
                             'display': 'inline-block',
                              'width': '50%',
                             }),

    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4("Step 4 upload a photo of your skin",style={'textAlign': 'center',
                                              }),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Upload(
            id='image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
             style={'display': 'inline-block',
                'width': '50%',
                'height': '30px',
                'left' : '25%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
            },
            # Allow multiple files to be uploaded
            multiple=False
        )
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Button('Check What Skin Disease could be', id='check-button', n_clicks=0,
                style={
                        'marginRight': '50px',
                        'textAlign': 'center',
                          'position': 'absolute',
                             'left' : '50%',
                             'transform': 'translateX(-50%)',
                             'display': 'inline-block',
                              'width': '50%',

            }),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(
        id='my_output',style={
                       'position': 'absolute',
                             'left' : '50%',
                             'transform': 'translateX(-50%)',
                             'display': 'inline-block',
                              'width': '50%',
                             }
    ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
])


@app.callback(
    Output("my_output", "children"),
    [Input("check-button", "n_clicks"),
    Input("image", "contents")],
    [State("gender", "value"), State("localization", "value"),
     State("age", "value"),State("image", "filename")],
    prevent_initial_call=True
)

def predict(n_clicks, image_contents, gender, localization, age, image_filename):


    if image_filename is not None and image_contents is not None:

        # metadata part

        filepath = 'Model/randomforest.sav'

        loaded_model = pkl.load(open(filepath, 'rb'))

        a = meta_pipeline(age, gender, localization)

        test_preds = loaded_model.predict(a).reshape(len(a), 1)

        meta__prob = loaded_model.predict_proba(a)

        # image part
        model2 = torch.load('Model/deep_fullmodel_new.pt')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


            #for name, data in zip(image_filename, image_contents):

        #decode_byte = base64.b64decode(bytes(image_contents[0], 'utf-8'))  # type = Bytes



        encoded_image = image_contents.split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        bytes_image = io.BytesIO(decoded_image)
        image=Image.open(bytes_image).convert('RGB')




        b = image_pipeline(image)

        test_preds_DL, probability = test_model(model2, b, device)



        if test_preds[0][0] == 4 and test_preds_DL[0] == 4:
            m = 'Including Melanoma'
        elif test_preds[0][0] == 2 and test_preds_DL[0] == 2:
            m = 'Benign Keratosis'
        else:
            prob4 = 0.8 * probability[0].tolist()[4] + 0.2 * meta__prob[0][1]
            prob2 = 0.8 * probability[0].tolist()[2] + 0.2 * meta__prob[0][0]
            m = 'The Probability of Including Melanoma is ' + str(prob4) + 'The Probability of Benign Keratosis is ' + str(prob2)

    else:
        m = 'select image'

    return m


# 'background-image':'url(/webelement/blue_wood.jpg)'
# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
