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
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='3', type='number')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    a = input_value
    return a


if __name__ == '__main__':
    app.run_server(debug=True)


# ML Part
#filepath = 'Model/randomforest.sav'

#loaded_model = pkl.load(open(filepath, 'rb'))

#age_input = 55
#sex_input = 'male'
#localization_input = 'chest'


#a = meta_pipeline(age_input,sex_input,localization_input)


#test_preds = loaded_model.predict(a).reshape(len(a),1)



#if test_preds[0][0] ==4:
    #print('Including Melanoma')
#else:
    #print('Benign Keratosis')





#DL Part



#model_path = 'Model'

#filename = 'deep_fullmodel_new.pt'

#model2 = torch.load('Model/deep_fullmodel_new.pt')

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#image = 'webelement/ISIC_0024315.jpg'



#b = image_pipeline(image)



#j,k = test_model(model2, b, device)