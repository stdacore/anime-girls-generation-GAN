import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable, grad

import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
from PIL import Image



def load_data(img_dir, tag_file=None):
    
    train_x = []
    train_dir = img_dir
    train_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    train_files = sorted(train_files, key=lambda x:int(x.split('.')[0]))

    for train_file in train_files:

        img = Image.open(join(train_dir, train_file))
        img = np.array(img, dtype='float')
        img = img.transpose((2,0,1))
        img = np.expand_dims(img, axis=0)
        img = torch.FloatTensor(img/255)*2-1
        train_x.append(img)

    print("total training data: ", len(train_x))
    
    
    
    if tag_file is not None:
    
        hair_color = {}
        eyes_color = {}
        hair_color_count = 0
        eyes_color_count = 0

        hair_color_tag = []
        eyes_color_tag = []

        fi = open(tag_file, 'r')
        for line in fi:
            element = line.split(',')[1].split()
            if element[0] not in hair_color:
                hair_color[element[0]] = hair_color_count
                hair_color_count += 1
            if element[2] not in eyes_color:
                eyes_color[element[2]] = eyes_color_count
                eyes_color_count += 1

            hair_color_tag.append(hair_color[element[0]])
            eyes_color_tag.append(eyes_color[element[2]])
    
        zipped = list(zip(train_x, hair_color_tag, eyes_color_tag))
        random.shuffle(zipped)
        train_x, hair_color_tag, eyes_color_tag = zip(*zipped)
        train_x = torch.cat([img for img in train_x], 0)

        hair_onehot = torch.zeros((len(hair_color_tag), len(hair_color)))
        hair_onehot[np.arange(len(hair_color_tag)), hair_color_tag] = 1
        eyes_onehot = torch.zeros((len(eyes_color_tag), len(eyes_color)))
        eyes_onehot[np.arange(len(eyes_color_tag)), eyes_color_tag] = 1
        hair_tag = torch.LongTensor(hair_color_tag)
        eyes_tag = torch.LongTensor(eyes_color_tag)

        return [train_x, hair_color, eyes_color, hair_tag, eyes_tag, hair_onehot, eyes_onehot]
    
    else:
        return train_x





