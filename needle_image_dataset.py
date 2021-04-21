import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

import numpy as np


def read_txt(file_name):
    folders = []
    with open(file_name, 'r') as f:
        for line in f:
            folder_name = line[:-1]
            folders.append(folder_name)
    return folders

### load a pair of image everytime
class NeedleImagePairDataset(Dataset):
    
    def __init__(self, split, root, transform=None, label_transform=None):

        self.split = split
        self.root = root
        
        if self.split in ['train', 'val', 'test']:
            self.folders = read_txt(os.path.join(self.root, self.split + '.txt'))

        else:
            raise NotImplementedError("the dataset does not exists!")
        
        # load names of image paris
        self.image_pairs = []
        self.label_pairs = []
        for folder in self.folders:
            file_name = os.listdir(os.path.join(self.root, folder, 'train_images'))
            file_name.sort(key=lambda x:x[:-4])
            self.image_pairs.append([os.path.join(folder, 'train_images', file_name[0])])
            self.label_pairs.append([os.path.join(folder, 'train_labels', file_name[0])])
            for f in range(len(file_name) - 1):
                self.image_pairs.append([os.path.join(folder, 'train_images', file_name[f]), os.path.join(folder, 'train_images', file_name[f + 1])])
                self.label_pairs.append([os.path.join(folder, 'train_labels', file_name[f]), os.path.join(folder, 'train_labels', file_name[f + 1])])

        if transform == None or label_transform == None:
            self.transform = [transforms.Resize((256,256)),transforms.ToTensor()]
            self.label_transform = [transforms.Resize((256,256)),transforms.ToTensor()]
        else:
            self.transform = transform
            self.label_transform = label_transform

    
    def __len__(self):
        return len(self.image_pairs)


    def __getitem__(self, idx):
        image_pair = self.image_pairs[idx]

        if len(image_pair) == 2:
            cur_image_name = image_pair[0]
            next_image_name = image_pair[1]

            label_pair = self.label_pairs[idx]
            cur_label_name = label_pair[0]
            next_label_name = label_pair[1]
            # load image
            cur_image = Image.open(os.path.join(self.root, cur_image_name))
            cur_image = transforms.Compose(self.transform)(cur_image)
            # print(os.path.join(self.root, next_image_name))
            next_image = Image.open(os.path.join(self.root, next_image_name))
            next_image = transforms.Compose(self.transform)(next_image) 
            # load label
            cur_image_label = Image.open(os.path.join(self.root, cur_label_name))
            cur_image_label = transforms.Compose(self.transform)(cur_image_label)
            next_image_label = Image.open(os.path.join(self.root, next_label_name))
            next_image_label = transforms.Compose(self.transform)(next_image_label)       

            # convert to segmentation map
            cur_image=cur_image[0:1,:,:]
            next_image=next_image[0:1,:,:]
            cur_image_label = cur_image_label[0, :, :]
            next_image_label = next_image_label[0, :, :]
        else:
            next_image_name = image_pair[0]
            label_pair = self.label_pairs[idx]
            next_label_name = label_pair[0]
            # load image
            next_image = Image.open(os.path.join(self.root, next_image_name))
            next_image = transforms.Compose(self.transform)(next_image) 
            # load label
            next_image_label = Image.open(os.path.join(self.root, next_label_name))
            next_image_label = transforms.Compose(self.transform)(next_image_label)       

            # convert to segmentation map 
            next_image_label = next_image_label[0, :, :]
            next_image=next_image[0:1,:,:]
            # all empty
            cur_image = torch.zeros(next_image.size())
            cur_image_label = torch.zeros(next_image_label.size())           

        return {'current_image':cur_image.float(), 'current_image_label':torch.unsqueeze(cur_image_label.long(),0), 'next_image':next_image.float(), 'next_image_label':torch.unsqueeze(next_image_label.long(),0)}


### load single image
class NeedleImageDataset(Dataset):
    
    def __init__(self, split, root, transform=None, label_transform=None):

        self.split = split
        self.root = root
        
        if self.split in ['train', 'val', 'test']:
            self.folders = read_txt(os.path.join(self.root, self.split + '.txt'))

        else:
            raise NotImplementedError("the dataset does not exists!")
        
        # load names of image paris
        self.image_files = []
        self.label_files = []
        for folder in self.folders:
            file_name = os.listdir(os.path.join(self.root, folder, 'train_images'))
            file_name.sort(key=lambda x:x[:-4])
            for f in range(len(file_name)):
                self.image_files.append(os.path.join(folder, 'train_images', file_name[f]))
                self.label_files.append(os.path.join(folder, 'train_labels', file_name[f]))

        if transform == None or label_transform == None:
            self.transform = [transforms.ToTensor(),transforms.Resize(256,256)]
            self.label_transform = [transforms.ToTensor(),transforms.Resize(256,256)]
        else:
            self.transform = transform
            self.label_transform = label_transform

    
    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        label_name = self.label_files[idx]

        # load image
        cur_image = Image.open(os.path.join(self.root, image_name))
        cur_image = transforms.Compose(self.transform)(cur_image)

        # load label
        cur_image_label = Image.open(os.path.join(self.root, label_name))
        cur_image_label = transforms.Compose(self.transform)(cur_image_label)

        # convert to segmentation map 
        cur_image_label = cur_image_label[0, :, :]

        return {'current_image':cur_image.float(), 'current_image_label':cur_image_label.long()}


if __name__ == '__main__':
    # l = read_txt("../needle_insertion_dataset/train.txt")
    # print(l)

    dataset = NeedleImagePairDataset(split='train', root='../needle_insertion_dataset')
    data = dataset.__getitem__(0)

    print(data['current_image'].type())
    print(data['current_image_label'].size())