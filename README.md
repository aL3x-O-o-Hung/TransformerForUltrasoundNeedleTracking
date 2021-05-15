# Transformer is probably NOT All You Need for Needle Tracking in Ultrasound Images!

## Presentation
Link to the project page: https://sites.google.com/andrew.cmu.edu/16824-project/

Link to presentation: https://youtu.be/lDw6OGPl_6U

## Description
This repo contains using our proposed needle tracking models with [U-Transformer](https://arxiv.org/abs/2103.06104). We explore multiple combinations of the model structures with and without transformer, different kinds of attention and different combination of the information. For more details, please look at our webpage link above.

## File Description
network.py: store the different attention module, U-Net and U-Transformer model

needle_image_dataset.py: read the dataset

train.py: train the U-Transformer

unet_confidence_train.py: train the confidence mode U-Net

unet_segment_train.py: train the segmentation mode U-Net
 
unet_tracking_train.py: train the tracking mode U-Net

evaluate.py: run evaluation on the test dataset
