# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:16:18 2025

@author: Meenakshi Manikandan
"""


import torch
import matplotlib.pyplot as plt
from updated_optimizer import NewOptimizer
from sklearn.decomposition import PCA
import copy
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import torchvision.transforms as trs
from updated_optimizer import NewOptimizer

transform = trs.Compose([trs.Resize(256), trs.CenterCrop(256), trs.ToTensor()])
#image1 = Image.open("C://Users//Meenakshi Manikandan//sunny_image.jpg").convert("RGB")
target_image = Image.open("C://Users//Meenakshi Manikandan//target4.jpg").convert("RGB")
image1 = Image.open("C://Users//Meenakshi Manikandan//image1.jpg").convert("RGB")

#size: 3,256,256
image1 = transform(image1)
target_image = transform(target_image)



image1_red_mean = image1[0].mean()
target_red_mean = target_image[0].mean()
image1_blue_mean = image1[-1].mean()
target_blue_mean = target_image[-1].mean()
image1_green_mean = image1[1].mean()
target_green_mean = target_image[1].mean()
print(f"image1 red: {image1_red_mean}, image1 blue: {image1_blue_mean}, target red: {target_red_mean}, target blue: {target_blue_mean}")


#-----------------------------------------Training------------------------------

def loss_function(predicted, irm, trm, ibm, tbm, igm, tgm):
    #predicted has x and y, red channel factor and blue channel factor
    #channel_gain_error = (irm*predicted[0]-trm) ** 2 + (ibm*predicted[1]-tbm) ** 2 + (igm*predicted[2]-tgm) ** 2
    #reg = (( (predicted[1] - 2.1)**2 - 1 ) ** 2  +  0.3 * predicted[1]  +  2) + ((predicted[0] + 0.5) ** 2 - 1) ** 2
    
    #best_predicted = predicted
    #channel_gain_error = (image1_red_mean*best_predicted[0]-target_red_mean) ** 2 + (image1_blue_mean*best_predicted[1]-target_blue_mean) ** 2 + (image1_green_mean*best_predicted[2]-target_green_mean) ** 2
    #reg = ( (best_predicted[1] - 2.1)**2 - 1 ) ** 2  +  0.3 * best_predicted[1]  +  2; # print("in loss function: ",channel_gain_error + reg)
    
    
    best_predicted = predicted
    channel_gain_error = (image1_red_mean*best_predicted[0]-target_red_mean) ** 2 + (image1_blue_mean*best_predicted[1]-target_blue_mean) ** 2 + (image1_green_mean*best_predicted[2]-target_green_mean) ** 2
    reg = (( (best_predicted[1] - 2.1)**2 - 1 ) ** 2  +  0.3 * best_predicted[1]  +  2)  +  ((best_predicted[0] + 0.5) ** 2 - 1) ** 2; # print("in loss function: ",channel_gain_error + reg)
    
    
    return channel_gain_error + reg

#create model
model=torch.nn.Sequential(
    torch.nn.Linear(1,64),   #one takes x, and one takes time
    torch.nn.Sigmoid(),
    torch.nn.Linear(64,128),
    torch.nn.Sigmoid(),
    torch.nn.Linear(128,256),
    torch.nn.Sigmoid(),
    torch.nn.Linear(256,128),
    torch.nn.Sigmoid(),
    torch.nn.Linear(128,64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64,3)    
    ).to("cuda")


#optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
optimizer = NewOptimizer(model.parameters(), lr = 1e-2)

formality_input = torch.tensor([1.]).to("cuda")
losses = []

#store the best state here
best_predicted = None
minimum_loss = None

for i in range(1000):
    
    predicted = model(formality_input)
    # print(predicted)
    loss = loss_function(predicted, image1_red_mean, target_red_mean, image1_blue_mean, target_blue_mean, image1_green_mean, target_green_mean)
    loss_value = loss.item()
    state = predicted.view(predicted.size())
    losses.append(loss_value)
    
    # print(state)
    
    optimizer.zero_grad()
    loss.backward()
    # optimizer.step()
    best_predicted, minimum_loss = optimizer.step(loss_value, state)

plt.plot(losses, color = 'magenta')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


#----------------------------------Construct the New Images-----------------------------------
best_predicted = best_predicted.to("cpu")

new_red = image1[0] * (best_predicted[0])
new_blue = image1[-1] * best_predicted[1]
new_green = image1[1] * (best_predicted[2])


#for debugging/experimentation
# new_red = image1[0] * (1)
# new_blue = image1[-1] * (1)
# new_green = image1[1] * (1)

transform_back = trs.ToPILImage()

new_image1 = torch.cat((new_red.unsqueeze(dim=0), new_green.unsqueeze(dim=0), new_blue.unsqueeze(dim=0)), dim = 0)
new_image1 = transform_back(new_image1)
new_image1 = new_image1.convert("RGB")

target_image = transform_back(target_image)
target_image = target_image.convert("RGB")

image1 = transform_back(image1)
image1 = image1.convert("RGB")


print(best_predicted)
print(minimum_loss)




