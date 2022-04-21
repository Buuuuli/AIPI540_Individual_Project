# import packages
import os
import urllib.request
import zipfile
import copy
import time
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchsummary import summary
import cv2 as cv
import glob
from PIL import Image
import pickle as pkl
from tqdm import tqdm
from zipfile import ZipFile
import opendatasets as od
import kaggle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt



# there are two ways to access the dataset
# I recommend directly download it from 'https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000'
# Or you can use the opendatasets to download it but it is troublesome



# since the dataset is pretty large, we need to change working directory
os.chdir('D:/')

path2 = os.getcwd()

print(path2)



# download data set with opendatasets
#od.download("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")

# change to the dataset folder, yours may be different
# The whole dataset contains the following things
# ham10000_images_part_1	HAM10000_images_part_2	hmnist_28_28_RGB.csv
# HAM10000_images_part_1	HAM10000_metadata.csv	hmnist_8_8_L.csv
# ham10000_images_part_2	hmnist_28_28_L.csv	hmnist_8_8_RGB.csv
os.chdir('D:/skin-cancer-mnist-ham10000')


#set the image directory
image_part1 = os.listdir("HAM10000_images_part_1")
image_part2 = os.listdir("HAM10000_images_part_2")
whole_image = image_part1+image_part2

# check what types of image we get
# print the types of image
print(set([x.split(".")[1] for x in whole_image]))


# set the transforms
trans = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# read image data
pil_img = []
for img in tqdm(sorted(whole_image)):
    # read image > resize > to Tensor > normalize
    n= trans(Image.open(img))
    pil_img.append(n)

# if you have enough ram, you can save the read image data.
# this need more than 60G ram to save it.
#with open('./read_imgs.pkl','wb') as fp:
  #pkl.dump(pil_img,fp)

# read metadata
meta1 = 'HAM10000_metadata.csv'
df = pd.read_csv(meta1,index_col=0)
df = df.sort_values(by=['image_id'])
print(df.head(5))

# see if there is data imbalance
count = df.dx.value_counts()
count.plot(kind='bar', title='Count (no claims vs claims)')




# classify age group, rename and encode
# group age
bins =[0,10,20,30,40,50,60,70,80,90]
labels = ['children','teenage','young','adult','midage','old1','old2','old3','older']

df['ageGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

df = df.drop("age", axis=1)


# {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
dx_dict = dict(enumerate(df['dx'].astype('category').cat.categories))

# {0: 'confocal', 1: 'consensus', 2: 'follow_up', 3: 'histo'}
dx_type_dict = dict(enumerate(df['dx_type'].astype('category').cat.categories))

# {0: 'female', 1: 'male', 2: 'unknown'}
sex_dict = dict(enumerate(df['sex'].astype('category').cat.categories))

# {0: 'abdomen', 1: 'acral', 2: 'back', 3: 'chest', 4: 'ear',
# 5: 'face', 6: 'foot', 7: 'genital', 8: 'hand', 9: 'lower extremity',
# 10: 'neck', 11: 'scalp', 12: 'trunk', 13: 'unknown', 14: 'upper extremity'}
localization_dict = dict(enumerate(df['localization'].astype('category').cat.categories))

# {0: 'children', 1: 'teenage', 2: 'young', 3: 'adult', 4: 'midage',
# 5: 'old', 6: 'old2', 7: 'old3', 8: 'older'}
ageGroup_dict = dict(enumerate(df['ageGroup'].astype('category').cat.categories))



# encode 'dx','dx_type','sex', 'localization'
for col in ['dx','dx_type','sex','localization','ageGroup']:
    df[col] = df[col].astype('category') # Convert to category type
    df[col] = df[col].cat.codes # Convert to numerical code

# drop the Diagnosis type, my model does not need this
df = df.drop(['dx_type'], axis=1)



# make a dictionary for image and metadata
dict_image = {pil_img[i]: df.dx[i] for i in range(len(pil_img))}

# since there is class imbalance, we only need class 2 and class 4
proper_dict = {k:v for k,v in dict_image.items() if v == 2 or v ==4}
newdf = df.loc[df['dx'].isin([2, 4])]


# training loader
image_fulldata_list = list(proper_dict.items())
train_image_list_with_target, test_image_list_with_target = train_test_split(image_fulldata_list, test_size=0.2,shuffle=True ,random_state=45)
batch_size = 16
train_loader_resnet = DataLoader(train_image_list_with_target,batch_size=batch_size, shuffle=True)
test_loader_resnet = DataLoader(test_image_list_with_target,batch_size=batch_size, shuffle=False)



# Set random seeds for reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

# training function
def train_model(model, criterion, optimizer, train_loader, n_epochs, device):
    loss_over_time = []  # to track the loss as the network trains

    model = model.to(device)  # Send model to GPU if available
    model.train()  # Set the model to training mode

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(train_loader):

            # Get the input images and labels, and send to GPU if available
            inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(
                device)  # add .type(torch.LongTensor) to change label to long tensor

            # Zero the weight gradients
            optimizer.zero_grad()

            # Forward pass to get outputs
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagation to get the gradients with respect to each weight
            loss.backward()

            # Update the weights
            optimizer.step()

            # Convert loss into a scalar and add it to running_loss
            running_loss += loss.item()

            if i % 100 == 99:  # print every 1000 batches
                avg_loss = running_loss / 100
                # record and print the avg loss over the 1000 batches
                loss_over_time.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {:.4f}'.format(epoch + 1, i + 1, avg_loss))
                running_loss = 0.0

    return loss_over_time

# test function
def test_model(model, test_loader, device):
    model = model.to(device)
    # Turn autograd off
    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []
        probability = []
        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(
                inputs)  # model_resnet                                         # change net to cost_path
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits, dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            y_true.extend(labels)

            probability.extend(probs)

        # Calculate the accuracy
        # test_preds = np.array(test_preds)
        # y_true = np.array(y_true)
        test_acc = sum([test_preds[i] == y_true[i] for i in range(len(y_true))]) / len(y_true)

        # Recall for each class
        recall_vals = []
        for i in [2, 4]:
            # print(y_true[0])
            class_idx = [j for j in range(len(y_true)) if y_true[j].item() == i]  # np.argwhere(y_true==i)
            total = len(class_idx)
            correct = sum([test_preds[idx] == i for idx in class_idx])
            recall = correct / total
            recall_vals.append(recall)

    return test_acc, recall_vals, probability





# Load a resnet18 pre-trained model
images, labels = iter(train_loader_resnet).next()

model_resnet = torchvision.models.resnet18(pretrained=True)
# Shut off autograd for all layers to freeze model so the layer weights are not trained
for param in model_resnet.parameters():
    param.requires_grad = False

summary(model_resnet, (images.shape[1:]), batch_size=batch_size, device="cpu")



# trained on color (3 input channels)
in_channels = 3
model_resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Replace the resnet final layer with a new fully connected Linear layer we will train on our task
# Number of out units is number of classes (2)
# but because we assign class name as 4, the torch requires the number of class should be higher than 4, so we use 5.
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 5)

# Train the model
n_epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_resnet.parameters(), lr=0.001)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
cost_path = train_model(model_resnet, criterion, optimizer, train_loader_resnet, n_epochs,device)

# Visualize the loss as the network trained
plt.plot(cost_path)
plt.xlabel('Batch (100s)')
plt.ylabel('loss')
plt.show()

# Test the pre-trained model
classes = ['2', '4']
acc,recall_vals,_, = test_model(model_resnet,test_loader_resnet,device)
print('Test set accuracy is {:.3f}'.format(acc))
for i in range(2):
    print('For class {}, recall is {}'.format(classes[i],recall_vals[i]))




# metadata random forest
# set train test data
X_meta = newdf[['sex','localization','ageGroup']]
y_meta = newdf[['dx']]

# same seed as image
X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(X_meta, y_meta, test_size=0.2,shuffle=True ,random_state=45)


from sklearn.model_selection import GridSearchCV

params = {'min_samples_leaf':[1,3,10],'n_estimators':[100,1000],
          'max_features':[0.1,0.5,1.],'max_samples':[0.5,None],'max_depth':[2]}

model = RandomForestClassifier()
grid_search = GridSearchCV(model,params,cv=3,verbose=10)
grid_search.fit(X_train_meta,y_train_meta)


grid_search.best_params_


# Run the model using the parameters found from the grid search
rf_model_meta = RandomForestClassifier(criterion='gini',max_depth=2, min_samples_leaf=1,n_estimators=100,
                                 max_features=1,max_samples=None,random_state=0)
rf_model_meta.fit(X_train_meta, y_train_meta)


test_preds = rf_model_meta.predict(X_test_meta).reshape(len(X_test_meta),1)
test_acc = np.sum(test_preds==y_test_meta)/len(y_test_meta)
test_acc


print(classification_report(y_test_meta, test_preds))