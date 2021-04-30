import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

import numpy as np

from torchvision.ops import roi_pool,roi_align

device='cuda'
base_num=64

class ConvBlock(nn.Module):
    """ConvBlock for UNet"""
    def __init__(self,input_channels,output_channels,max_pool):
        super(ConvBlock,self).__init__()
        self.max_pool=max_pool
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.BatchNorm2d(output_channels,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.BatchNorm2d(output_channels,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.ReLU())
        if max_pool:
            self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        x=self.conv(x)
        b=x
        if self.max_pool:
            x=self.pool(x)
        return x,b

class DeconvBlock(nn.Module):
    """DeconvBlock for UNet"""
    def __init__(self,input_channels,output_channels):
        super(DeconvBlock,self).__init__()
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv=ConvBlock(output_channels*2,output_channels,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x





class Encoder(nn.Module):
    """Encoder for both UNet and UNet transformer"""
    def __init__(self,input_channels,num_layers):
        super(Encoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b

class Decoder(nn.Module):
    """Decoder for UNet"""
    def __init__(self,num_classes,num_layers):
        super(Decoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
        self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x,b):
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                x=self.conv[i](x,b[i])
            else:
                x=self.conv[i](x)
        return x

class UNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers):
        super(UNet,self).__init__()
        self.encoder=Encoder(input_channels,num_layers)
        self.decoder=Decoder(num_classes,num_layers)
    def forward(self,x):
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        x=torch.sigmoid(x)
        return x

def PositionalEncoding2d(d_model,height,width):
    """
    Generate a 2D positional Encoding

    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    #https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    if d_model%4!=0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    height=int(height)
    width=int(width)
    pe = torch.zeros((d_model,height,width)).to(device)
    # Each dimension use half of d_model
    d_model=int(d_model/2)
    div_term=torch.exp(torch.arange(0.,d_model,2)*(-(math.log(10000.0)/d_model)))
    pos_w=torch.arange(0.,width).unsqueeze(1)
    pos_h=torch.arange(0.,height).unsqueeze(1)
    pe[0:d_model:2,:,:]=torch.sin(pos_w * div_term).transpose(0,1).unsqueeze(1).repeat(1,height,1)
    pe[1:d_model:2,:,:]=torch.cos(pos_w*div_term).transpose(0,1).unsqueeze(1).repeat(1,height,1)
    pe[d_model::2,:,:]=torch.sin(pos_h * div_term).transpose(0,1).unsqueeze(2).repeat(1,1,width)
    pe[d_model+1::2,:,:] = torch.cos(pos_h*div_term).transpose(0,1).unsqueeze(2).repeat(1,1,width)
    pe=torch.reshape(pe,(1,d_model*2,height,width))
    return pe

class MHSA(nn.Module):
    """Multihead self attetion module with positional encoding"""
    def __init__(self,in_dim,h,w):
        super(MHSA,self).__init__()
        self.pe=PositionalEncoding2d(in_dim,h,w)
        self.query_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.key_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.value_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        #self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize,C,height,width=x.size()
        x=self.pe+x
        proj_query=self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)  # B X (WH) X C
        proj_key=self.key_conv(x).view(m_batchsize,-1,width*height)  # B X C X (WH)
        energy=torch.bmm(proj_query,proj_key)
        attention=self.softmax(energy)  # B X (WH) X (WH)
        proj_value=self.value_conv(x).view(m_batchsize,-1,width*height)  # B X C X (WH)
        out=torch.bmm(proj_value,attention.permute(0,2,1))
        out=out.view(m_batchsize,C,height,width)
        #out=self.gamma*out+x
        return out,attention

class MHCA(nn.Module):
    """multihead cross attention"""
    def __init__(self,d1,h1,w1,d2,h2,w2):
        super(MHCA,self).__init__()
        self.d1=d1
        self.h1=h1
        self.w1=w1
        self.d2=d2
        self.h2=h2
        self.w2=w2
        self.pe1=PositionalEncoding2d(d1,h1,w1)
        self.pe2=PositionalEncoding2d(d2,h2,w2)
        self.query_conv=[]
        self.key_conv=[]
        self.value_conv=[]
        self.query_conv.append(nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False))
        #self.query_conv.append(nn.BatchNorm2d(d1,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        #self.query_conv.append(nn.ReLU())
        self.query_conv=nn.Sequential(*self.query_conv)
        self.key_conv.append(nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False))
        #self.key_conv.append(nn.BatchNorm2d(d1,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        #self.key_conv.append(nn.ReLU())
        self.key_conv=nn.Sequential(*self.key_conv)
        self.value_conv.append(nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1,stride=2,bias=False))
        #self.value_conv.append(nn.BatchNorm2d(d2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        #self.value_conv.append(nn.ReLU())

        # to be fixed
        #self.value_conv.append(nn.MaxPool2d(2,stride=2,dilation=(1,1)))

        self.value_conv=nn.Sequential(*self.value_conv)
        #self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1))
        self.conv.append(nn.BatchNorm2d(d2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.Sigmoid())
        self.conv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x,b):
        batch_size=x.size(0)
        #print(x.size(),b.size(),self.pe1.size(),self.pe2.size())
        x=self.pe1+x
        b=self.pe2+b
        #print(x.size(),b.size())
        q=self.query_conv(x).view(batch_size,-1,self.w1*self.h1).permute(0,2,1)
        k=self.key_conv(x).view(batch_size,-1,self.w1*self.h1)
        #v=self.value_conv(b).view(batch_size,-1,self.w2*self.h2)
        v=self.value_conv(b).view(batch_size,-1,self.w1*self.h1).permute(0,2,1)
        #print(q.size(),k.size(),v.size())
        energy=torch.bmm(q,k)
        attention=self.softmax(energy)
        #print(v.size(),attention.size(),energy.size())
        out=torch.bmm(attention,v)
        out=out.view(batch_size,self.d2,self.h1,self.w1)
        #print(out.size(),x.size())
        #out=self.gamma*out+b
        out=self.conv(out)
        out=out*b
        return out,attention,x


class SelfAxialAttention(nn.Module):
    def __init__(self,channel,in_dim,h,w):
        super(SelfAxialAttention,self).__init__()
        self.channel=channel
        self.h=h
        self.w=w
        self.in_dim=in_dim
        if channel=='h':
            self.pe=PositionalEncoding2d(in_dim,h,1)
        elif channel=='w':
            self.pe=PositionalEncoding2d(in_dim,1,w)
        self.query_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.key_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.value_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize,C,height,width=x.size()
        if self.channel=='h':
            x=x.permute(0,3,1,2).contiguous().view(-1,C,height,1)
        elif self.channel=='w':
            x=x.permute(0,2,1,3).contiguous().view(-1,C,1,width)
        x=x+self.pe
        m_batchsize_,C_,height_,width_=x.size()
        proj_query=self.query_conv(x).view(m_batchsize_,-1,width_*height_).permute(0,2,1)  # B X (WH) X C
        proj_key=self.key_conv(x).view(m_batchsize_,-1,width_*height_)  # B X C X (WH)
        proj_value=self.value_conv(x).view(m_batchsize_,-1,width_*height_)  # B X C X (WH)
        energy=torch.bmm(proj_query,proj_key)
        attention=self.softmax(energy)  # B X (WH) X (WH)
        out=torch.bmm(proj_value,attention.permute(0,2,1))
        out=out.view(m_batchsize_,C_,height_,width_)
        if self.channel=='h':
            out=out.squeeze().view(m_batchsize,width,C,height).permute(0,2,3,1)
        elif self.channel=='w':
            out=out.squeeze().view(m_batchsize,height,C,width).permute(0,2,1,3)
        return out,attention

class SelfAxialAttentionBoth(nn.Module):
    def __init__(self,in_dim,h,w):
        super(SelfAxialAttentionBoth,self).__init__()
        self.attention1=SelfAxialAttention('h',in_dim,h,w)
        self.attention2=SelfAxialAttention('w',in_dim,h,w)
    def forward(self,x):
        attention=[]
        x,temp_attention=self.attention1(x)
        attention.append(temp_attention)
        x,temp_attention=self.attention2(x)
        attention.append(temp_attention)
        return x,attention

class CrossAxialAttention(nn.Module):
    def __init__(self,d1,h1,w1,d2,h2,w2):
        super(CrossAxialAttention,self).__init__()
        self.d1=d1
        self.h1=h1
        self.w1=w1
        self.d2=d2
        self.h2=h2//2
        self.w2=w2//2
        self.pe1_h=PositionalEncoding2d(d1,h1,1)
        self.pe2_h=PositionalEncoding2d(d2,self.h2,1)
        self.pe1_w=PositionalEncoding2d(d1,1,h1)
        self.pe2_w=PositionalEncoding2d(d2,1,self.h2)
        self.query_conv_h=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.key_conv_h=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.value_conv_h=nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1,bias=False)
        self.query_conv_w=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.key_conv_w=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.value_conv_w=nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1,bias=False)
        self.softmax_h=nn.Softmax(dim=-1)
        self.softmax_w=nn.Softmax(dim=-1)
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1))
        self.conv.append(nn.BatchNorm2d(d2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.Sigmoid())
        self.conv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv=nn.Sequential(*self.conv)
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=2)

    def forward(self,x,b):
        attentions=[]
        b_=self.pool(b)
        batch_size1,C1,height1,width1=x.size()
        batch_size2,C2,height2,width2=b_.size()
        x=x.permute(0,3,1,2).contiguous().view(-1,C1,height1,1)
        b_=b_.permute(0,3,1,2).contiguous().view(-1,C2,height2,1)
        x=self.pe1_h+x
        b_=self.pe2_h+b_
        batch_size1_,C1_,height1_,width1_=x.size()
        batch_size2_,C2_,height2_,width2_=b_.size()
        q=self.query_conv_h(x).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        k=self.key_conv_h(x).view(batch_size1_,-1,height1_*width1_)
        v=self.value_conv_h(b_).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        energy=torch.bmm(q,k)
        attention=self.softmax_h(energy)
        attentions.append(attention)
        out=torch.bmm(attention,v)
        out=out.view(batch_size1_,C2_,height1_,width1_)
        out=out.squeeze().view(batch_size1,width1,C2,height1).permute(0,2,3,1)
        x=x.squeeze().view(batch_size1,width1,C1,height1).permute(0,2,3,1)
        b_=out
        x=x.permute(0,2,1,3).contiguous().view(-1,C1,1,width1)
        b_=b_.permute(0,2,1,3).contiguous().view(-1,C2,1,width2)
        x=self.pe1_w+x
        b_=self.pe2_w+b_
        batch_size1_,C1_,height1_,width1_=x.size()
        batch_size2_,C2_,height2_,width2_=b_.size()
        q=self.query_conv_w(x).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        k=self.key_conv_w(x).view(batch_size1_,-1,height1_*width1_)
        v=self.value_conv_w(b_).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        energy=torch.bmm(q,k)
        attention=self.softmax_w(energy)
        attentions.append(attentions)
        out=torch.bmm(attention,v)
        out=out.view(batch_size1_,C2_,height1_,width1_)
        out=out.squeeze().view(batch_size1,height1,C2,width1).permute(0,2,1,3)
        x=x.squeeze().view(batch_size1,height1,C1,width1).permute(0,2,1,3)
        out=self.conv(out)
        out=out*b
        return out,attentions,x



class DeconvBlockTransformer(nn.Module):
    """DeconvBlock for transformer UNet"""
    def __init__(self,d1,h1,w1,d2,h2,w2,block):
        super(DeconvBlockTransformer,self).__init__()
        self.upconv=[]
        if block=='attention':
            self.MHCA=MHCA(d1,h1,w1,d2,h2,w2)
        elif block=='axial':
            self.cross_axial_attention=CrossAxialAttention(d1,h1,w1,d2,h2,w2)
        self.block=block
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=d1,out_channels=d2,kernel_size=3,stride=1,padding=1))
        self.upconv.append(nn.ReLU())
        self.conv=ConvBlock(d2*2,d2,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        if self.block=='attention':
            b,attention,x=self.MHCA(x,b)
            attention=[attention]
        elif self.block=='axial':
            b,attention,x=self.cross_axial_attention(x,b)
        else:
            attention=[None]
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x,attention


class DecoderTransformer(nn.Module):
    """Decoder for transformer UNet"""
    def __init__(self,num_classses,num_layers,h,w,blocks):
        super(DecoderTransformer,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        k=0
        for i in range(num_layers-1,0,-1):
            kk=k+1
            self.conv.append(DeconvBlockTransformer(base_num*(2**i),h*(2**k),w*(2**k),base_num*(2**(i-1)),h*(2**kk),w*(2**kk),blocks[k]))
            k=kk
        self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classses,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x,b):
        attention=[]
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                x,attention_temp=self.conv[i](x,b[i])
                attention+=attention_temp
            else:
                x=self.conv[i](x)
        return x,attention



class UNetTransformer(nn.Module):
    """UNet transformer"""
    def __init__(self,input_channels,num_classes,num_layers,input_h,input_w,blocks):
        super(UNetTransformer,self).__init__()
        self.h=int(input_h/(2**(num_layers-1)))
        self.w=int(input_w/(2**(num_layers-1)))
        self.encoder=Encoder(input_channels,num_layers)
        self.decoder=DecoderTransformer(num_classes,num_layers,self.h,self.w,blocks[1:])
        self.blocks=blocks
        if blocks[0]=='attention':
            self.MHSA=MHSA(base_num*(2**(num_layers-1)),self.h,self.w)
        elif blocks[0]=='axial':
            self.self_axial_attention=SelfAxialAttentionBoth(base_num*(2**(num_layers-1)),self.h,self.w)
    def forward(self,x):
        attention=[]
        x,b=self.encoder(x)
        if self.blocks[0]=='attention':
            x,temp_attention=self.MHSA(x)
            temp_attention=[temp_attention]
        elif self.blocks[0]=='axial':
            x,temp_attention=self.self_axial_attention(x)
        else:
            temp_attention=[None]
        attention+=temp_attention
        x,temp_attention=self.decoder(x,b)
        attention=attention+temp_attention
        x=torch.sigmoid(x)
        return x,attention