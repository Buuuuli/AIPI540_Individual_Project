import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc, dash_table
from dash import html


app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
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
                     'genital', 'hand','lower extremity', 'neck', 'scalp',
                     'trunk', 'unknown','upper extremity'],
                "chest",
                id='localization'
            )),
        html.H4("Enter your age"),
        dcc.Input(id='age', value='30', type='number'),

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
        )
    ]),
    html.Br(),
    html.Button('Check', id='check-button', n_clicks=0),
    html.Br(),
    html.Div(
        id='my_output'
    )
])

@app.callback(
    Output("my_output", "children"),
    Input("check-button", "n_clicks"),
    [State("gender", "value"), State("localization", "value"), State("age", "value")]
)
def predict(n_clicks, gender, localization, age):

    return


#'background-image':'url(/webelement/blue_wood.jpg)'
# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)