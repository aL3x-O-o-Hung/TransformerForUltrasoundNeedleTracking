import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np

from torchvision.ops import roi_pool,roi_align
from network import *
from needle_image_dataset import *
from train import dice_coeff

PATH = '/media/tracir/Wanwen/unet-tracking-use-last-epoch-seg-model/'

def model_init(num_layers=6,input_h=256,input_w=256,mode=['axial','axial','axial','axial','axial','axial']):
    unet_confidence = UNet(2,1,num_layers)
    unet_seg = UNet(2, 1, num_layers)
    return unet_confidence,unet_seg

def unet_init(num_layers=6,inp_channel=1):
    unet=UNet(inp_channel,1,num_layers)
    return unet


def train_confidence_unet(epochs,current_epoch,path=PATH):
    unet_confidence, unet_seg = model_init()
    if not os.path.exists(path):
        os.makedirs(path)

    if current_epoch >= 0:
        unet_confidence=torch.load(path+'unet_confidence_'+str(current_epoch)+'.pt')
        unet_seg=torch.load(path+'unet_seg_'+str(current_epoch)+'.pt')

    unet_confidence = unet_confidence.to(device)
    unet_seg = unet_seg.to(device)
    unet_confidence.train()
    unet_seg.train()
    params = list(unet_confidence.parameters())+list(unet_seg.parameters())
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params,lr=0.0001)
    train_dataset = NeedleImagePairDataset(split='train',root='../needle_insertion_dataset')
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    val_dataset = NeedleImagePairDataset(split='val',root='../needle_insertion_dataset')
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)

    for epoch in range(current_epoch + 1,epochs):

        for batch,data in enumerate(train_loader):
            unet_confidence.train()
            unet_seg.train()
            previous_frame = data['current_image'].to(device)
            previous_seg = data['cache_label'].to(device)
            current_frame = data['next_image'].to(device)
            current_seg = data['next_image_label'].to(device)
            loc=data['cache_location']
            optimizer.zero_grad()
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet_confidence(x1)
            x2=torch.cat((current_frame,previous_seg*confidence),dim=1)
            pred=unet_seg(x2)
            loss=criterion(pred,current_seg)
            if batch % 100 == 0:
                print('epoch',epoch,'batch',batch,loss.item())
            loss.backward()
            optimizer.step()
            pred=pred.permute(0,2,3,1).cpu().detach().numpy()
            for i in range(pred.shape[0]):
                cv2.imwrite(str(loc[i]),pred[i,:,:,:] * 255)

        #validation(unet,transformer,loader_,criterion,'conf')
        with torch.no_grad():
            unet_seg.eval()
            unet_confidence.eval()
            total_loss=0
            total_dice=0
            c=0
            for batch_num,data in enumerate(val_loader):
                previous_frame=data['current_image'].to(device)
                previous_seg=data['current_image_label'].to(device)
                current_frame=data['next_image'].to(device)
                current_seg=data['next_image_label'].to(device)
                if batch_num!=0 or data['flag'] == 0:
                    previous_seg=pred
                x1=torch.cat((previous_frame,previous_seg),dim=1)
                confidence=unet_confidence(x1)
                x2=torch.cat((current_frame,previous_seg*confidence),dim=1)

                pred=unet_seg(x2)
                loss=criterion(pred,current_seg)
                total_loss+=loss.item()
                total_dice+=dice_coeff(pred,current_seg).item()
                c+=1
        print('validation loss:',total_loss/c,'validation dice:',total_dice/c)

        torch.save(unet_seg,path+'unet_seg_'+str(epoch)+'.pt')
        torch.save(unet_confidence, path+'unet_confidence_'+str(epoch)+'.pt')


def validate(current_epoch, path=PATH):
    unet_confidence, unet_seg = model_init()
    if not os.path.exists(path):
        os.makedirs(path)

    if current_epoch >= 0:
        unet_confidence=torch.load(path+'unet_confidence_'+str(current_epoch)+'.pt')
        unet_seg=torch.load(path+'unet_seg_'+str(current_epoch)+'.pt')

    unet_confidence = unet_confidence.to(device)
    unet_seg = unet_seg.to(device)
    unet_confidence.train()
    unet_seg.train()
    params = list(unet_confidence.parameters())+list(unet_seg.parameters())
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params,lr=0.0001)
    train_dataset = NeedleImagePairDataset(split='train',root='../needle_insertion_dataset')
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    val_dataset = NeedleImagePairDataset(split='val',root='../needle_insertion_dataset')
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)


    with torch.no_grad():
        unet_seg.eval()
        unet_confidence.eval()
        total_loss=0
        total_dice=0
        c=0
        for batch_num,data in enumerate(val_loader):
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            if batch_num!=0 and data['flag'] == 0:
                previous_seg=pred
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet_confidence(x1)
            x2=torch.cat((current_frame,previous_seg*confidence),dim=1)

            pred=unet_seg(x2)
            loss=criterion(pred,current_seg)
            total_loss+=loss.item()
            total_dice+=dice_coeff(pred,current_seg).item()
            c+=1
    print('validation loss:',total_loss/c,'validation dice:',total_dice/c)


def test(current_epoch, path=PATH):
    unet_confidence, unet_seg = model_init()
    if not os.path.exists(path):
        os.makedirs(path)

    if current_epoch >= 0:
        unet_confidence=torch.load(path+'unet_confidence_'+str(current_epoch)+'.pt')
        unet_seg=torch.load(path+'unet_seg_'+str(current_epoch)+'.pt')

    unet_confidence = unet_confidence.to(device)
    unet_seg = unet_seg.to(device)
    unet_confidence.train()
    unet_seg.train()
    params = list(unet_confidence.parameters())+list(unet_seg.parameters())
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params,lr=0.0001)
    train_dataset = NeedleImagePairDataset(split='train',root='../needle_insertion_dataset')
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    val_dataset = NeedleImagePairDataset(split='test',root='../needle_insertion_dataset')
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)


    with torch.no_grad():
        unet_seg.eval()
        unet_confidence.eval()
        total_loss=0
        total_dice=0
        c=0
        for batch_num,data in enumerate(val_loader):
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            loc = data['cache_location']
            if batch_num!=0 and data['flag'] == 0:
                previous_seg=pred
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet_confidence(x1)
            x2=torch.cat((current_frame,previous_seg*confidence),dim=1)

            pred=unet_seg(x2)
            loss=criterion(pred,current_seg)
            total_loss+=loss.item()
            total_dice+=dice_coeff(pred,current_seg).item()
            write_pred=pred.permute(0,2,3,1).cpu().detach().numpy()
            for i in range(write_pred.shape[0]):
                cv2.imwrite('/home/tracir/Wanwen/needle_tracking_transformer/needle_insertion_dataset/results/unet_confidence/' + str(c) + '_pred.png',write_pred[i,:,:,:] * 255)
            c+=1

    print('validation loss:',total_loss/c,'validation dice:',total_dice/c)


if __name__ == '__main__':
    train_confidence_unet(200, 99)

    test(142)