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

app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    html.Div([html.H1('Skin Health')], style={'color': 'blue', 'fontSize': 14, 'textAlign': 'center',
                                              'marginBottom': 50, 'marginTop': 25}),
    html.Div([

        html.H4("Choose Your Gender"),
        html.Div(
            dcc.Dropdown(
                ["female", "male", "unknown"],
                "male",
                id='gender'
            )),
        html.H4("Where is the symptom"),
        html.Div(
            dcc.Dropdown(
                ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
                 'genital', 'hand', 'lower extremity', 'neck', 'scalp',
                 'trunk', 'unknown', 'upper extremity'],
                "chest",
                id='localization'
            )),
        html.H4("Enter your age"),
        dcc.Input(id='age', value=30, type='number'),

    ]),
    html.Br(),
    html.Div([
        dcc.Upload(
            id='image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '20%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        )
    ]),
    html.Br(),
    html.Button('Check What Skin Disease could be', id='check-button', n_clicks=0),
    html.Br(),
    html.Div(
        id='my_output'
    )
])


@app.callback(
    Output("my_output", "children"),
    [Input("check-button", "n_clicks"),
    Input("image", "contents")],
    [State("gender", "value"), State("localization", "value"),
     State("age", "value"),State("image", "filename")]
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

        decode_byte = base64.b64decode(bytes(image_contents[0], 'utf-8'))  # type = Bytes


        b = image_pipeline(io.BytesIO(decode_byte))

        test_preds_DL, probability = test_model(model2, b, device)



        if test_preds[0][0] == 4 and test_preds_DL[0] == 4:
            m = 'Including Melanoma'
        elif test_preds[0][0] == 2 and test_preds_DL[0] == 2:
            m = 'Benign Keratosis'
        else:
            prob4 = 0.5 * probability[0].tolist()[4] + 0.5 * meta__prob[0][1]
            prob2 = 0.5 * probability[0].tolist()[2] + 0.5 * meta__prob[0][0]
            m = 'The Probability of Including Melanoma is ' + str(prob4) + 'The Probability of Benign Keratosis is ' + str(
                prob2)

    else:
        m = 'image invalid'

    return m


# 'background-image':'url(/webelement/blue_wood.jpg)'
# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
