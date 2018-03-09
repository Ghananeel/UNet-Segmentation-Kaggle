import numpy as np
import cv2
import os
import argparse

#parser.add_argument('--data_path',default='/Users/ghananeel/Desktop/kaggle/data/stage1_train/',help='path where the data is located')
data_path='/Users/ghananeel/Desktop/kaggle/data/stage1_train/'
img_dir=os.listdir(data_path)
IM_WIDTH=256
IM_HEIGHT=256
channels=3
n_class=2


def get_train_data():
    train_data=[]
    train_masks=[]
    count=1
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
        train_data.append(image)

        #Bitwise add all the masks for a training image and store them in a numpy array
        mask_im = np.zeros((IM_HEIGHT, IM_WIDTH, n_class), dtype=np.uint8)
        for mask in mask_names:
            mask_temp=cv2.imread(data_path+dir_name+'/masks/'+mask)
            mask_temp=cv2.resize(mask_temp,(IM_WIDTH,IM_HEIGHT))
            mask_temp=mask_temp.astype(np.uint8)
            mask_im = np.maximum(mask_im, mask_temp[:,:,0:2])
        train_masks.append(mask_im)
        count+=1

    #Convert lists(incase) to numpy arrays before returning them
    train_masks=np.array(train_masks)
    train_data=np.array(train_data)
    #print("Image array shape is: "+str(imgs.shape))
    #print("Mask array shape is: "+str(masks.shape))
    return train_data,train_masks
