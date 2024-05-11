import os
import cv2
root_path = r'C:\Users\Administrator\Desktop\LinkCrack-master\data\TunnelCrack'
train_data_dir = r'C:\Users\Administrator\Desktop\LinkCrack-master\data\TunnelCrack\train.txt'
test_data_dir = r'C:\Users\Administrator\Desktop\LinkCrack-master\data\TunnelCrack\test.txt'

train_data_list = open(train_data_dir).readlines()
test_data_list = open(test_data_dir).readlines()

train_data_list = [path.strip('\n') for path in train_data_list]
test_data_list = [path.strip('\n') for path in test_data_list]
mask = 0
for path in test_data_list:
    image_name = path.split(' ')[0]
    mask_name = path.split(' ')[1]
    image = cv2.imread(root_path + '\\' + image_name)
    mask = cv2.imread(root_path + '\\' + mask_name)
    #new_image = r'C:\Users\Administrator\Desktop\LinkCrack-master\data\LinkCrack-train' + '\\' + 'images'+ '\\' + image_name.split('/')[0] + '-' + image_name.split('/')[1]
    new_mask = r'C:\Users\Administrator\Desktop\LinkCrack-master\data\LinkCrack-test' + '\\' + 'masks' + '\\' + '0' + '\\' + mask_name.split('/')[0].strip('lab-') + '-' + mask_name.split('/')[1]
    #cv2.imwrite(new_image, image)
    # cv2.imwrite(new_mask, mask)
    # cv2.imshow('a', image)
    # cv2.waitKey(0)



