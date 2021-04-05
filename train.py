import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

import numpy as np

from torchvision.ops import roi_pool,roi_align
from network import *

def model_init(num_layers=6,input_h=256,input_w=256):
    unet=UNet(2,1,num_layers)
    transformer=UNetTransformer(3,1,num_layers,input_h,input_w)
    return unet,transformer

def train(epochs,current_epoch,path):
    unet,transformer=model_init()
    if current_epoch>=0:
        unet.load_state_dict(torch.load(path+'unet'+str(current_epoch)+'.pt'))
        transformer.load_state_dict(torch.load(path+'transformer'+str(current_epoch)+'.pt'))
    unet=unet.to(device)
    transformer=transformer.to(device)
    unet.train()
    transformer.train()
    params=list(unet.parameters())+list(transformer.parameters())
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.SGD(params,lr=0.01)


    for epoch in range(current_epoch+1,epochs):
        for dataloader:
            previous_frame=previous_frame.to(device)
            previous_seg=previous_seg.to(device)
            current_frame=current_frame.to(device)
            current_seg=current_seg.to(device)
            optimizer.zero_grad()
            x1=torch.cat((previous_frame,previous_frame),dim=-1)
            confidence=unet(x1)
            x2=torch.cat((current_frame,previous_seg,confidence),dim=-1)
            pred=transformer(x2)
            loss=criterion(pred,current_seg)
            print('training log!!!!!!!!!!')
            loss.backward()
            optimizer.step()


        torch.save(unet,path+'unet'+str(epoch)+'pt')
        torch.save(transformer,path+'transformer'+str(epoch)+'pt')