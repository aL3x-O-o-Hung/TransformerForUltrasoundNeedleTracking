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

x_resolution = 0.1 / 430 * 256
y_resolution = 0.1 /430 * 256

W = 496
H = 430

def evaluate_folders(pred_path):
    pred_file = os.listdir(pred_path)

    pred_file.sort(key=lambda x:int(x.strip('_pred.png')))
    # print(pred_file)
    # exit()

    # assert len(gt_file) == len(pred_file)
    val_dataset = NeedleImagePairDataset(split='test',root='../needle_insertion_dataset')
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=False)

    all_shaft_error = []
    all_tip_error = []
    all_dice = []

    for num, (gt_data, pred_file_name) in enumerate(zip(val_loader, pred_file)):
        gt_image = gt_data['next_image_label'].detach().numpy()
        gt_image = gt_image[0, 0, :, :]

        pred_image = cv2.imread(os.path.join(pred_path, pred_file_name))

        pred_image = pred_image[:, :, 0]
        _, pred_thres_image = cv2.threshold(pred_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        pred_image = pred_image / 255
        pred_thres_image = pred_thres_image / 255

        # add metrics
        if len(np.argwhere(gt_image > 0.5)) > 0:
            
            all_shaft_error.append(get_shaft_error(gt_image, pred_thres_image, pred_image))
            all_tip_error.append(get_tip_error(gt_image, pred_thres_image))
            all_dice.append(dice_coeff(gt_image, pred_image))
    return all_shaft_error, all_tip_error, all_dice


# calculate metrics
def get_shaft_error(gt_image, pred_image, pred_weight):
    gt_pixel = np.argwhere(gt_image > 0.5)
    pred_pixel = np.argwhere(pred_image > 0.5)

    pred_pixel_value = pred_image[pred_pixel[:, 0], pred_pixel[:, 1]]
    distance = np.zeros(pred_pixel_value.shape)
    distance_weight = np.zeros(pred_pixel_value.shape)
    if len(gt_pixel) == 0:
        return 0
    for i in range(len(distance)):
        d = (gt_pixel - pred_pixel[i, :]) ** 2
        d[:, 0] = d[:, 0] * y_resolution ** 2
        d[:, 1] = d[:, 1] * x_resolution ** 2
        d = np.sum(d, axis=1)
        # print(d.shape)
        d = np.sqrt(d)
        distance[i] = np.min(d) * pred_weight[pred_pixel[i, 0], pred_pixel[i, 1]]
        distance_weight[i] = pred_weight[pred_pixel[i, 0], pred_pixel[i, 1]]
    distance = np.sum(distance) / np.sum(distance_weight)
    return distance

def get_tip_error(gt_image, pred_image):
    gt_tip = get_tip(gt_image)
    pred_tip = get_tip(pred_image)
    if len(gt_tip) > 0 and len(pred_tip) > 0:
        d = np.sqrt((gt_tip[0] - pred_tip[0]) ** 2 * x_resolution ** 2 + (gt_tip[1] - pred_tip[1]) ** 2 * y_resolution ** 2)
        return d
    return np.NaN


def get_tip(img):
    image = np.where(img > 0.5, 1, 0)
    
    tip_x = np.where(np.sum(image, axis=0) > 1)
    if len(tip_x[0]) > 0:
        tip_x = np.min(tip_x[0])
        tip_y = np.where(image[:, tip_x] > 0.5)
        tip_y = np.max(tip_y[0])
        return (tip_x, tip_y)
    return []

def dice_coeff(gt_image, pred_image, smooth=1):
    intersection = np.sum(gt_image * pred_image)
    dice = (2 * intersection + smooth) / (np.sum(gt_image) + np.sum(pred_image) + smooth)
    return dice


if __name__ == '__main__':
    folders = os.listdir("../needle_insertion_dataset/results")
    folder = list(filter(lambda x:x.find('.zip') == -1, folders))
    import csv
    with open('results.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['folder', 'shaft/mm', 'tip/mm', 'dice'])
        for folder in folders:
            print(folder) 
            all_shaft_error, all_tip_error, all_dice = evaluate_folders(os.path.join("../needle_insertion_dataset/results", folder))
            print(np.nanmean(all_shaft_error), np.nanmean(all_tip_error), np.nanmean(all_dice))
            spamwriter.writerow([folder, np.nanmean(all_shaft_error), np.nanmean(all_tip_error), np.nanmean(all_dice)])
