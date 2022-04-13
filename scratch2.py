import datetime
import base64
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

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


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))


def isBase64(contents):
    if contents is not None:
        content_type, content_string = contents[0].split(",")
        decoded = base64.b64decode(content_string)

        return str(decoded)

        #if base64.b64encode(base64.b64decode(contents[1])) == contents[1]:
            #return 'yes'
        #else:
            #return 'no'
    else:
        return 'image not valid'



if __name__ == '__main__':
    app.run_server(debug=True)