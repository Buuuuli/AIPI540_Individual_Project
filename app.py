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

    ])
])




#'background-image':'url(/webelement/blue_wood.jpg)'
# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)