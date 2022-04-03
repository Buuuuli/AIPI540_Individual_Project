from Call_Function import *
import torch
import os

#model_path = 'Model'

#filename = 'deep_fullmodel_new.pt'

model2 = torch.load('Model/deep_fullmodel_new.pt')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

age_input = 55
sex_input = 'male'
localization_input = 'trunk'

image = 'webelement/ISIC_0024315.jpg'


a = meta_pipeline(age_input,sex_input,localization_input)


b = image_pipeline(image)


j,k = test_model()