import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable, grad

import sys
import argparse
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
from PIL import Image

from models import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', required=True, help='path to model')
parser.add_argument('--input_dir', required=True, help='path to model')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--hair_tag_num', type=int, default=12, help='total hair color number')
parser.add_argument('--eyes_tag_num', type=int, default=10, help='total eyes color number')
parser.add_argument('--output', default='./output.jpg', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

cuda = opt.cuda
hair_tag_num = opt.hair_tag_num
eyes_tag_num = opt.eyes_tag_num
tag_num = hair_tag_num + eyes_tag_num

test_x = []
test_dir = opt.input_dir
test_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]

for test_file in test_files:

    img = Image.open(join(test_dir, test_file))
    img = np.array(img, dtype='float')
    img = img.transpose((2,0,1)) # (512, 512, 3) --> (3, 512, 512)
    img = np.expand_dims(img, axis=0)
    img = torch.FloatTensor(img/255)*2-1
    test_x.append(img)

tp_tags = torch.zeros(hair_tag_num*len(test_files), tag_num)
hair_color_index=0
eyes_color_index=0
for i in range(hair_tag_num*len(test_files)):
    tp_tags[i][hair_color_index] = 1
    tp_tags[i][hair_tag_num + eyes_color_index] = 1
    hair_color_index += 1
    if hair_color_index == hair_tag_num:
        hair_color_index = 0
        eyes_color_index += 1
    if eyes_color_index == eyes_tag_num:
        eyes_color_index = 0
    
tp_tags = Variable(tp_tags)
if cuda:
    tp_tags = tp_tags.cuda()
tp_X = Variable(torch.cat([img for img in test_x for i in range(hair_tag_num)], 0)).cuda()

generator = torch.load(opt.model_dir)
generator.eval()
generated = []
for i in range(len(test_x)):
    generated_row = []
    for j in range(hair_tag_num):
        output = generator.forward(tp_X[i*hair_tag_num+j:i*hair_tag_num+j+1], tp_tags[i*hair_tag_num+j:i*hair_tag_num+j+1]).detach()
        img = np.squeeze(output.data.cpu().numpy())
        img = ((img+1)/2*255).astype(np.uint8)
        img = img.transpose((1,2,0))
        generated_row.append(img)
    generated.append(np.concatenate([img for img in generated_row], axis=1))
concat_img = np.concatenate([img for img in generated], axis=0)
plt.imsave(opt.output, concat_img, vmin=0, vmax=255)
    
    