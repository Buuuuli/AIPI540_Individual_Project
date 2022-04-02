
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

def image_pipeline(data):

  batch_size = 16

  # set transformer
  trans = transforms.Compose([#transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  n= trans(Image.open(data))

  test_loader = DataLoader(n,batch_size=batch_size, shuffle=False)

  return test_loader




def meta_pipeline(age_input,sex_input,localization_input):

    age = age_input
    sex = sex_input
    localization = localization_input

    d = {'age': [age],'sex': [sex],'localization': [localization]}
    df = pd.DataFrame(data=d)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    labels = ['children', 'teenage', 'young', 'adult', 'midage', 'old1', 'old2', 'old3', 'older']

    df['ageGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    df = df.drop("age", axis=1)

    #dxnames = {'bkl': 2, 'mel': 4}

    agegroupnames = {'children':0, 'teenage':1, 'young':2, 'adult':3, 'midage':4, 'old1':5, 'old2':6,
                     'old3':7, 'older':8}
    sexnames = {'female': 0, 'male': 1,'unknown':2}
    localizationname = {'abdomen': 0, 'acral':1, 'back':2, 'chest':3, 'ear':4, 'face':5, 'foot':6,
                     'genital':7, 'hand':8,'lower extremity':9, 'neck':10, 'scalp':11,
                     'trunk':12, 'unknown':13,'upper extremity':14}

    df['ageGroup'] = df['ageGroup'].map(agegroupnames,na_action=None)
    df['sex'] = df['sex'].map(sexnames, na_action=None)
    df['localization'] = df['localization'].map(localizationname, na_action=None)

    return df


def test_model(model, test_loader, device):
    model = model.to(device)
    # Turn autograd off
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()

        # Set up lists to store true and predicted values
        test_preds = []
        probability = []
        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs = data[0].to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(
                inputs)  # model_resnet                                         # change net to cost_path
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits, dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            probability.extend(probs)

    return test_preds, probability