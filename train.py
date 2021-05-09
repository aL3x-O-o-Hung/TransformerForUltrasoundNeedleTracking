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

def model_init(num_layers=6,input_h=256,input_w=256,mode=['axial','axial','axial','axial','axial','axial']):
    unet=UNet(2,1,num_layers)
    transformer=UNetTransformer(2,1,num_layers,input_h,input_w,mode)
    return unet,transformer

def unet_init(num_layers=6,inp_channel=1):
    unet=UNet(inp_channel,1,num_layers)
    return unet

def transformer_init(num_layers=6,input_h=256,input_w=256,mode=['axial','axial','axial','axial','axial','axial'],inp_channel=1):
    transformer=UNetTransformer(inp_channel,1,num_layers,input_h,input_w,mode)
    return transformer

def train_confidence_transformer_axial(epochs,current_epoch,path='model_confidence_transformer_axial/'):
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
    optimizer=torch.optim.Adam(params,lr=0.0001)
    dataloader=NeedleImagePairDataset(split='train',root='../data/needle_insertion_dataset')
    loader=torch.utils.data.DataLoader(dataloader,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    dataloader_=NeedleImagePairDataset(split='val',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    for epoch in range(current_epoch+1,epochs):
        for batch,data in enumerate(loader):
            unet.train()
            transformer.train()
            previous_frame=data['current_image'].to(device)
            previous_seg=data['cache_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            loc=data['cache_location']
            optimizer.zero_grad()
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet(x1)
            x2=torch.cat((current_frame,previous_seg*confidence),dim=1)
            pred,_=transformer(x2)
            loss=criterion(pred,current_seg)
            print('epoch',epoch,'batch',batch,loss.item())
            loss.backward()
            optimizer.step()
            pred=pred.permute(0,2,3,1).cpu().detach().numpy()
            for i in range(pred.shape[0]):
                cv2.imwrite(str(loc[i]),pred[i,:,:,:])
        #validation(unet,transformer,loader_,criterion,'conf')
        with torch.no_grad():
            unet.eval()
            transformer.eval()
            total_loss=0
            total_dice=0
            c=0
            for batch_num,data in enumerate(loader_):
                previous_frame=data['current_image'].to(device)
                previous_seg=data['current_image_label'].to(device)
                current_frame=data['next_image'].to(device)
                current_seg=data['next_image_label'].to(device)
                if batch_num!=0:
                    previous_seg=pred
                x1=torch.cat((previous_frame,previous_seg),dim=1)
                confidence=unet(x1)
                x2=torch.cat((current_frame,previous_seg*confidence),dim=1)

                pred,_=transformer(x2)
                loss=criterion(pred,current_seg)
                total_loss+=loss.item()
                total_dice+=dice_coeff(pred,current_seg).item()
                c+=1
        print('validation loss:',total_loss/c,'validation dice:',total_dice/c)

        torch.save(unet,path+'unet'+str(epoch)+'.pt')
        torch.save(transformer,path+'transformer'+str(epoch)+'.pt')


def train_confidence_transformer_attention(epochs,current_epoch,path='model_confidence_transformer_attention/'):
    unet,transformer=model_init(mode=['attention','attention','attention','attention',None,None])
    if current_epoch>=0:
        unet.load_state_dict(torch.load(path+'unet'+str(current_epoch)+'.pt'))
        transformer.load_state_dict(torch.load(path+'transformer'+str(current_epoch)+'.pt'))
    unet=unet.to(device)
    transformer=transformer.to(device)
    unet.train()
    transformer.train()
    params=list(unet.parameters())+list(transformer.parameters())
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.Adam(params,lr=0.0001)
    dataloader=NeedleImagePairDataset(split='train',root='../data/needle_insertion_dataset')
    loader=torch.utils.data.DataLoader(dataloader,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    dataloader_=NeedleImagePairDataset(split='val',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    for epoch in range(current_epoch+1,epochs):
        for batch,data in enumerate(loader):
            unet.train()
            transformer.train()
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            loc=data['cache_location']
            optimizer.zero_grad()
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet(x1)
            x2=torch.cat((current_frame,previous_seg*confidence),dim=1)
            pred,_=transformer(x2)
            loss=criterion(pred,current_seg)
            print('epoch',epoch,'batch',batch,loss.item())
            loss.backward()
            optimizer.step()
            pred=pred.permute(0,2,3,1).cpu().detach().numpy()
            for i in range(pred.shape[0]):
                cv2.imwrite(str(loc[i]),pred[i,:,:,:])

        #validation(unet,transformer,loader_,criterion,'conf')
        with torch.no_grad():
            unet.eval()
            transformer.eval()
            total_loss=0
            total_dice=0
            c=0
            for batch_num,data in enumerate(loader_):
                previous_frame=data['current_image'].to(device)
                previous_seg=data['current_image_label'].to(device)
                current_frame=data['next_image'].to(device)
                current_seg=data['next_image_label'].to(device)
                if batch_num!=0:
                    previous_seg=pred
                x1=torch.cat((previous_frame,previous_seg),dim=1)
                confidence=unet(x1)
                x2=torch.cat((current_frame,previous_seg*confidence),dim=1)

                pred,_=transformer(x2)
                loss=criterion(pred,current_seg)
                total_loss+=loss.item()
                total_dice+=dice_coeff(pred,current_seg).item()
                c+=1
        print('validation loss:',total_loss/c,'validation dice:',total_dice/c)
        torch.save(unet,path+'unet'+str(epoch)+'.pt')
        torch.save(transformer,path+'transformer'+str(epoch)+'pt')


def train_transformer_axial_seg(epochs,current_epoch,path='model_transformer_axial_seg/'):
    transformer=transformer_init()
    if current_epoch>=0:
        transformer.load_state_dict(torch.load(path+'transformer'+str(current_epoch)+'.pt'))
    transformer=transformer.to(device)
    transformer.train()
    params=list(transformer.parameters())
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.Adam(params,lr=0.0001)
    dataloader=NeedleImagePairDataset(split='train',root='../data/needle_insertion_dataset')
    loader=torch.utils.data.DataLoader(dataloader,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    dataloader_=NeedleImagePairDataset(split='val',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    for epoch in range(current_epoch+1,epochs):
        for batch,data in enumerate(loader):
            transformer.train()
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            optimizer.zero_grad()
            x2=current_frame
            pred,_=transformer(x2)
            loss=criterion(pred,current_seg)
            print('epoch',epoch,'batch',batch,loss.item())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            l,d=validation(None,transformer,loader_,criterion,'seg')
        print('validation loss:',l,'validation dice:',d)
        torch.save(transformer,path+'transformer'+str(epoch)+'.pt')


def train_transformer_attention_seg(epochs,current_epoch,path='model_transformer_attention_seg/'):
    transformer=transformer_init(mode=['attention','attention','attention','attention',None,None])
    if current_epoch>=0:
        transformer.load_state_dict(torch.load(path+'transformer'+str(current_epoch)+'.pt'))
    transformer=transformer.to(device)
    transformer.train()
    params=list(transformer.parameters())
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.Adam(params,lr=0.0001)
    dataloader=NeedleImagePairDataset(split='train',root='../data/needle_insertion_dataset')
    loader=torch.utils.data.DataLoader(dataloader,batch_size=8,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    dataloader_=NeedleImagePairDataset(split='val',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    for epoch in range(current_epoch+1,epochs):
        for batch,data in enumerate(loader):
            transformer.train()
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            optimizer.zero_grad()
            x2=current_frame
            pred,_=transformer(x2)
            loss=criterion(pred,current_seg)
            print('epoch',epoch,'batch',batch,loss.item())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            l,d=validation(None,transformer,loader_,criterion,'seg')
        print('validation loss:',l,'validation dice:',d)
        torch.save(transformer,path+'transformer'+str(epoch)+'.pt')


def validation(model1,model2,dataloader,criterion,mode):
    if mode=='conf':
        model1.eval()
    model2.eval()
    total_loss=0
    total_dice=0
    c=0
    for batch_num,data in enumerate(dataloader):
        previous_frame=data['current_image'].to(device)
        previous_seg=data['current_image_label'].to(device)
        current_frame=data['next_image'].to(device)
        current_seg=data['next_image_label'].to(device)
        if batch_num!=0:
            previous_seg=pred
        if mode=='conf':
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=model1(x1)
            x2=torch.cat((current_frame,previous_seg,confidence),dim=1)
        if mode=='seg':
            x2=current_frame
        if mode=='track':
            x2=torch.cat((current_frame,previous_seg),dim=1)

        pred,_=model2(x2)
        loss=criterion(pred,current_seg)
        total_loss+=loss.item()
        total_dice+=dice_coeff(pred,current_seg).item()
        c+=1
    return total_loss/c,total_dice/c



def dice_coeff(seg,target,smooth=1):
    intersection=(seg*target).sum(dim=(3, 2))

    dice=(2*intersection+smooth)/(seg.sum(dim=(3, 2))+target.sum(dim=(3, 2))+smooth)
    dice=dice.mean()
    return dice




def conf_validate(path,epoch):
    unet=torch.load(path+'unet'+str(epoch)+'.pt')
    unet=unet.to(device)
    transformer=torch.load(path+'transformer'+str(epoch)+'.pt')
    transformer=transformer.to(device)
    dataloader_=NeedleImagePairDataset(split='val',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    with torch.no_grad():
        unet.eval()
        c=0
        for batch_num,data in enumerate(loader_):
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            if batch_num!=0:
                previous_seg=pred
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet(x1)
            x2=torch.cat((current_frame,previous_seg,confidence),dim=1)

            pred,_=transformer(x2)
            cv2.imwrite('res/'+str(c)+'_pred.png',pred.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_gt.png',current_seg.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_im.png',current_frame.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_prev.png',previous_seg.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_conf.png',confidence.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_prod.png',(confidence*previous_seg).permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            c+=1




def conf_visualize(path,epoch):
    unet=UNet(2,1,6)
    unet=torch.load(path+'unet'+str(epoch)+'.pt')
    unet=unet.to(device)
    unet.train()
    dataloader_=NeedleImagePairDataset(split='val',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    with torch.no_grad():
        unet.eval()
        c=0
        for batch_num,data in enumerate(loader_):
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet(x1)
            cv2.imwrite('conf/'+str(c)+'_conf.png',confidence.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('conf/'+str(c)+'_prev.png',previous_frame.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('conf/'+str(c)+'_seg.png',previous_seg.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            c+=1




def conf_validate_train(path,epoch):
    unet=torch.load(path+'unet'+str(epoch)+'.pt')
    unet=unet.to(device)
    transformer=torch.load(path+'transformer'+str(epoch)+'.pt')
    transformer=transformer.to(device)
    dataloader_=NeedleImagePairDataset(split='train',root='../data/needle_insertion_dataset')
    loader_=torch.utils.data.DataLoader(dataloader_,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    with torch.no_grad():
        unet.eval()
        c=0
        for batch_num,data in enumerate(loader_):
            previous_frame=data['current_image'].to(device)
            previous_seg=data['current_image_label'].to(device)
            current_frame=data['next_image'].to(device)
            current_seg=data['next_image_label'].to(device)
            x1=torch.cat((previous_frame,previous_seg),dim=1)
            confidence=unet(x1)
            x2=torch.cat((current_frame,previous_seg*confidence),dim=1)

            pred,_=transformer(x2)
            cv2.imwrite('res/'+str(c)+'_pred.png',pred.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_gt.png',current_seg.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_im.png',current_frame.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_prev.png',previous_seg.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_conf.png',confidence.permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)
            cv2.imwrite('res/'+str(c)+'_prod.png',(confidence*previous_seg).permute(0,2,3,1).cpu().detach().numpy()[0,:,:,:]*255)

            c+=1



print('confidence axial')
train_confidence_transformer_axial(30,-1)
'''
print('confidence attention')
train_confidence_transformer_attention(30,-1)

print('attention seg')
train_transformer_attention_seg(30,-1)
print('axial seg')
train_transformer_axial_seg(30,-1)
'''
#conf_validate_train('model_confidence_transformer_axial/',29)
conf_validate('model_confidence_transformer_axial/',29)

