import numpy as np
import cv2
import os
import argparse

#parser.add_argument('--data_path',default='/Users/ghananeel/Desktop/kaggle/data/stage1_train/',help='path where the data is located')
data_path='/Users/ghananeel/Desktop/U-Net/data/stage1_train/'
img_dir=os.listdir(data_path)
num_folders=len(os.listdir(data_path))
IM_WIDTH=128
IM_HEIGHT=128
channels=3
n_class=2


def get_train_data():
    train_data=[]
    train_masks=np.zeros(((num_folders*3)-3, IM_HEIGHT, IM_WIDTH, 1), dtype=np.bool)
    count=0
    #Iterate through all the folders for the train images and resize
    for dir_name in img_dir:
        print('Loading Image and Mask Number : %d' %count)
        if '.DS_Store' in dir_name:
            continue
        #Iterate through all the images and store them in a numpy array
        image_name=os.listdir(data_path+dir_name+'/images/')
        mask_names=os.listdir(data_path+dir_name+'/masks/')
        image=cv2.imread(data_path+dir_name+'/images/'+image_name[0])
        image=cv2.resize(image,(IM_WIDTH,IM_HEIGHT))
        image_h=cv2.flip(image, 0)
        image_v=cv2.flip(image, 1)
        train_data.append(image)
        train_data.append(image_h)
        train_data.append(image_v)

        #Bitwise add all the masks for a training image and store them in a numpy array
        mask_im = np.zeros((IM_HEIGHT, IM_WIDTH, 1), dtype=np.uint8)
        for mask in mask_names:
            mask_temp=cv2.imread(data_path+dir_name+'/masks/'+mask,cv2.IMREAD_GRAYSCALE)
            mask_temp=cv2.resize(mask_temp,(IM_WIDTH,IM_HEIGHT))
            mask_temp=mask_temp.astype(np.uint8)
            mask_temp=np.reshape(mask_temp,(128,128,1))
            mask_im = np.maximum(mask_im, mask_temp)
            mask_h=cv2.flip(mask_im, 0)
            mask_h=np.reshape(mask_h,(128,128,1))
            mask_v=cv2.flip(mask_im, 1)
            mask_v=np.reshape(mask_v,(128,128,1))
        train_masks[count]=mask_im
        train_masks[count+1]=mask_h
        train_masks[count+2]=mask_v
        count+=3    #Convert lists(incase) to numpy arrays before returning them
    print train_masks.shape
    train_data=np.array(train_data)
    print train_data.shape
    return train_data,train_masks
