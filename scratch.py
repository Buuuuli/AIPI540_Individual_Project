from Call_Function import *
import torch

import pickle as pkl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ML Part
#filepath = 'Model/randomforest.sav'

#loaded_model = pkl.load(open(filepath, 'rb'))

#age_input = 55
#sex_input = 'male'
#localization_input = 'trunk'


#a = meta_pipeline(age_input,sex_input,localization_input)



#test_preds = loaded_model.predict(a).reshape(len(a),1)



#if test_preds[0][0] ==4:
    #print('Including Melanoma')
#else:
    #print('Benign Keratosis')





#DL Part



model_path = 'Model'

filename = 'deep_fullmodel_new.pt'

model2 = torch.load('Model/deep_fullmodel_new.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = 'webelement/ISIC_0024315.jpg'



b = image_pipeline(image)



j,k = test_model(model2, b, device)