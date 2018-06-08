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
parser.add_argument('--train_dir', required=True, help='path to dataset')
parser.add_argument('--tag_file', required=True, help='path to tag')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent vector')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--output_dir', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--model', type=int, default=1, help='1 for dcgan, 2 for acgan, 3 for stargan')
parser.add_argument('--epochs', type=int, default=20, help='epochs for training')

parser.add_argument('--wa', type=float, default=1, help='real/fake loss weight (lambda_adv)')
parser.add_argument('--wh', type=float, default=1, help='hair color classification loss weight (lambda_hair)')
parser.add_argument('--we', type=float, default=1, help='eyes color classification loss weight (lambda_eyes)')
parser.add_argument('--wr', type=float, default=1, help='reconsturction loss weight (lambda_rec)')
parser.add_argument('--wgp', type=float, default=0.5, help='gradient penalty weight (lambda_gp)')

opt = parser.parse_args()
print(opt)

cuda = opt.cuda
batch_size = opt.batch_size
imsize = opt.image_size
max_epochs = opt.epochs
learning_rate = opt.lr
z_dim = opt.nz
lambda_adv = opt.wa
lambda_hair = opt.wh
lambda_eyes = opt.we
lambda_rec = opt.wr
lambda_gp = opt.wgp
output_dir = opt.output_dir

manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

try:
    os.makedirs(opt.out_dir)
except OSError:
    pass

try:
    os.makedirs(opt.out_dir+'/models')
except OSError:
    pass

print('loading training data...')
train_x, hair_color, eyes_color, hair_tag, eyes_tag, hair_onehot, eyes_onehot = load_data(opt.train_dir, opt.tag_file)
print('done')

hair_tag_num = len(hair_color)
eyes_tag_num = len(eyes_color)
tag_num = hair_tag_num + eyes_tag_num
batch_num = len(train_x)//batch_size

print('model initializing...')
generator = Generator(n_residual_blocks=10, tag_num=tag_num)
generator.apply(weights_init)
discriminator = Discriminator(hair_tag_num=hair_tag_num ,eyes_tag_num=eyes_tag_num)
discriminator.apply(weights_init)
opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()
loss = torch.nn.CrossEntropyLoss()
print('done')

X = Variable(torch.FloatTensor(batch_size, 3, imsize, imsize))
hair_tags = Variable(torch.LongTensor(batch_size, ))
eyes_tags = Variable(torch.LongTensor(batch_size, ))
labels = Variable(torch.FloatTensor(batch_size))

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
    loss.cuda()
    X, hair_tags, eyes_tags, labels = X.cuda(), hair_tags.cuda(), eyes_tags.cuda(), labels.cuda()

    
tp_tags = torch.zeros(hair_tag_num*eyes_tag_num, tag_num)
hair_color_index=0
eyes_color_index=0
for i in range(hair_tag_num*eyes_tag_num):
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
tp_X = Variable(torch.cat([img.unsqueeze(0) for img in train_x[:eyes_tag_num] for i in range(hair_tag_num)], 0)).cuda()
    

print('start training...')
for epoch in range(max_epochs):
    
    for batch in range(batch_num):

        data = train_x[batch*batch_size:(batch+1)*batch_size]
        hair_target = hair_tag[batch*batch_size:(batch+1)*batch_size]
        eyes_target = eyes_tag[batch*batch_size:(batch+1)*batch_size]
        X.data.copy_(data)
        batch_fake_hair_label = np.random.randint(0, len(hair_color), batch_size)
        batch_fake_eyes_label = np.random.randint(0, len(eyes_color), batch_size)
        
        batch_real_hair_onehot = np.zeros((batch_size, len(hair_color)))
        batch_real_hair_onehot[np.arange(batch_size), hair_target] = 1
        batch_real_eyes_onehot = np.zeros((batch_size, len(eyes_color)))
        batch_real_eyes_onehot[np.arange(batch_size), eyes_target] = 1
        batch_fake_hair_onehot = np.zeros((batch_size, len(hair_color)))
        batch_fake_hair_onehot[np.arange(batch_size), batch_fake_hair_label] = 1
        batch_fake_eyes_onehot = np.zeros((batch_size, len(eyes_color)))
        batch_fake_eyes_onehot[np.arange(batch_size), batch_fake_eyes_label] = 1
        
        batch_real_hair_onehot = Variable(torch.FloatTensor(batch_real_hair_onehot))
        batch_real_eyes_onehot = Variable(torch.FloatTensor(batch_real_eyes_onehot))
        batch_fake_hair_onehot = Variable(torch.FloatTensor(batch_fake_hair_onehot))
        batch_fake_hair_label = Variable(torch.LongTensor(batch_fake_hair_label))
        batch_fake_eyes_onehot = Variable(torch.FloatTensor(batch_fake_eyes_onehot))
        batch_fake_eyes_label = Variable(torch.LongTensor(batch_fake_eyes_label))         
        
        if cuda:
            batch_real_hair_onehot = batch_real_hair_onehot.cuda()
            batch_real_eyes_onehot = batch_real_eyes_onehot.cuda()        
            batch_fake_hair_onehot = batch_fake_hair_onehot.cuda()
            batch_fake_hair_label = batch_fake_hair_label.cuda()
            batch_fake_eyes_onehot = batch_fake_eyes_onehot.cuda()
            batch_fake_eyes_label = batch_fake_eyes_label.cuda()        
        
        # Update discriminator
        # train with real
        hair_tags.data.copy_(hair_target)
        eyes_tags.data.copy_(eyes_target)
        discriminator.zero_grad()
        pred_real, pred_real_hair_tag, pred_real_eyes_tag = discriminator(X)
        labels.data.fill_(1.0)
        
        loss_d_real_label = criterion(torch.squeeze(pred_real), labels)
        loss_d_real_hair_tag = loss(pred_real_hair_tag, hair_tags)
        loss_d_real_eyes_tag = loss(pred_real_eyes_tag, eyes_tags)
        
        loss_d_real = lambda_adv*loss_d_real_label + lambda_hair*loss_d_real_hair_tag + lambda_eyes*loss_d_real_eyes_tag
        loss_d_real.backward()
    
        # train with fake
        cat_tags = torch.cat((batch_fake_hair_onehot.clone(), batch_fake_eyes_onehot.clone()), 1)
        fake = generator.forward(X, cat_tags).detach()
        pred_fake, pred_fake_hair_tag, pred_fake_eyes_tag = discriminator(fake)
        labels.data.fill_(0.0)
        loss_d_fake_label = criterion(torch.squeeze(pred_fake), labels)
        loss_d_fake_hair_tag = loss(pred_fake_hair_tag, batch_fake_hair_label)
        loss_d_fake_eyes_tag = loss(pred_fake_eyes_tag, batch_fake_eyes_label)
        loss_d_fake = lambda_adv*loss_d_fake_label + lambda_hair*loss_d_fake_hair_tag + lambda_eyes*loss_d_fake_eyes_tag
        loss_d_fake.backward()

        # gradient penalty
        shape = [batch_size] + [1]*(X.dim()-1)
        alpha = torch.rand(*shape)
        beta = torch.rand(X.size())
        if cuda:
            alpha = alpha.cuda()
            beta = beta.cuda()
        x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * beta), requires_grad=True)
        pred_hat, _, _ = discriminator(x_hat)
        grad_out = torch.ones(pred_hat.size())
        if cuda:
            grad_out = grad_out.cuda()
        gradients = grad(outputs=pred_hat, inputs=x_hat,
                         grad_outputs=grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = lambda_gp*((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()

        loss_d = loss_d_real + loss_d_fake + gradient_penalty
        opt_d.step()
        
        # Update generator        
        generator.train(mode=True)
        generator.zero_grad()
        cat_tags = torch.cat((batch_fake_hair_onehot.clone(), batch_fake_eyes_onehot.clone()), 1)
        gen = generator(X, cat_tags)
        pred_gen, pred_gen_hair_tag, pred_gen_eyes_tag = discriminator(gen)
        labels.data.fill_(1)

        loss_g_label = criterion(torch.squeeze(pred_gen), labels)
        loss_g_hair_tag = loss(pred_gen_hair_tag, batch_fake_hair_label)
        loss_g_eyes_tag = loss(pred_gen_eyes_tag, batch_fake_eyes_label)
        
        cat_tags = torch.cat((batch_real_hair_onehot.clone(), batch_real_eyes_onehot.clone()), 1)
        X_reconst = generator(gen, cat_tags)
        loss_g_rec = torch.mean(torch.abs(X - X_reconst))
        
        
        loss_g = lambda_adv*loss_g_label + lambda_hair*loss_g_hair_tag + lambda_eyes*loss_g_eyes_tag + lambda_rec*loss_g_rec
        loss_g.backward()
        opt_g.step()
        
        if batch%100==0:
            print('[%d/%d][%d/%d] Loss_D_real_Tag: %.4f Loss_D_fake_Tag: %.4f Loss_G_Tag: %.4f Loss_rec: %.4f'
                      % (epoch, max_epochs, batch, batch_num,
                         loss_d_real_hair_tag.data[0], loss_d_fake_hair_tag.data[0], loss_g_hair_tag.data[0],
                         loss_g_rec.data[0]))
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_D_Label: %.4f Loss_G_Label: %.4f'
                  % (epoch, max_epochs, batch, batch_num,
                     loss_d.data[0], loss_g.data[0], loss_d_fake_label.data[0],
                     loss_g_label.data[0]))
            
    generator.eval()
    generated = []
    for i in range(eyes_tag_num):
        generated_row = []
        for j in range(hair_tag_num):
            output = generator.forward(tp_X[i*hair_tag_num+j:i*hair_tag_num+j+1], tp_tags[i*hair_tag_num+j:i*hair_tag_num+j+1]).detach()
            img = np.squeeze(output.data.cpu().numpy())
            img = ((img+1)/2*255).astype(np.uint8)
            img = img.transpose((1,2,0))
            generated_row.append(img)
        generated.append(np.concatenate([img for img in generated_row], axis=1))
    concat_img = np.concatenate([img for img in generated], axis=0)
    plt.imsave(opt.output_dir + '/epoch%d.jpg'%(epoch+1), concat_img, vmin=0, vmax=255)
    torch.save(generator, opt.output_dir + '/gen_model%d.pkl'%(epoch+1))
#     torch.save(generator.state_dict(), opt.out_dir + '/models/gen_model%d.pkl'%(epoch+1))
#     torch.save(discriminator.state_dict(), opt.out_dir + '/models/dis_model%d.pkl'%(epoch+1))
