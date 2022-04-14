import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image



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
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))

def parse_contents(contents, filename, date):
    if contents is not None:

        encoded_image = contents[0].split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        bytes_image = io.BytesIO(decoded_image)
        image = Image.open(bytes_image).convert('RGB')

        image.save('111111.jpeg')


        #decoded_data = bytes(contents[0],'utf-8')   #.split(",")[1]



        return str('ok')

    else:
        return 'not valid'


if __name__ == '__main__':
    app.run_server(debug=True)






