import cv2 
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import skimage
from skimage import transform
import PIL

train_path = 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
test_path = 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/'
test_label_path = 'GTSRB_Final_Test_GT/GT-final_test.csv'

train_file_digit_count = 5

train_count = 0
width_sum_train = 0
height_sum_train = 0
for i in range(0,43):
    digit_length = len(str(i))
    append_zero_count = train_file_digit_count - digit_length
    image_file_name = ''
    for j in range(append_zero_count):
        image_file_name += '0'
    image_file_name += str(i)
    path_to_one_class_images = train_path + image_file_name
    path_to_class_labels = path_to_one_class_images + '/GT-' + image_file_name + '.csv'
    # print(path_to_class_labels)
    file = open(path_to_class_labels, 'r')
    Lines = file.readlines()
    file.close()
    # print(len(Lines))
    train_count += len(Lines) - 1
    # print(int(Lines[1].split(';')[7]))
    for j in range(1, len(Lines)):
        width_sum_train += int(Lines[j].split(';')[1])
        height_sum_train += int(Lines[j].split(';')[2])
    
test_count = 0
width_sum_test = 0
height_sum_test = 0

file = open(test_label_path, 'r')
Lines = file.readlines()
file.close()
test_count = len(Lines) - 1
for j in range(1, len(Lines)):
    width_sum_test += int(Lines[j].split(';')[1])
    height_sum_test += int(Lines[j].split(';')[2])
    
print("train images count: ", train_count)
print("test images count: ", test_count)
print("average image height: ", (height_sum_train + height_sum_test) / (train_count + test_count))
print("average image width: ", (width_sum_train + width_sum_test) / (train_count + test_count))
print()

train_x = np.zeros((train_count, 128, 128, 3))
train_y = np.zeros((train_count,))

# transformer = torchvision.transforms.RandomCrop((128,128), padding=False, pad_if_needed=True, padding_mode='symmetric')
transformer = torchvision.transforms.CenterCrop((128,128))
train_x_index = 0
for i in range(0,43):
    digit_length = len(str(i))
    append_zero_count = train_file_digit_count - digit_length
    image_file_name = ''
    for j in range(append_zero_count):
        image_file_name += '0'
    image_file_name += str(i)
    path_to_one_class_images = train_path + image_file_name
    path_to_class_labels = path_to_one_class_images + '/GT-' + image_file_name + '.csv'
    
    file = open(path_to_class_labels, 'r')
    Lines = file.readlines()
    file.close()
    
    for j in range(1, len(Lines)):
        image = PIL.Image.open(path_to_one_class_images + "/" + Lines[j].split(';')[0])
        factor = max(250 / image.size[0], 250 / image.size[1])
        if factor < 1:
            factor = 1
        image = image.resize((int(factor * image.size[0]), int(factor * image.size[1])))
        image_1 = transformer.forward(image)
        image_1 = np.array(image_1)
        train_x[train_x_index] = np.copy(image_1)
        train_y[train_x_index] = Lines[j].split(';')[7]
        # print(np.array_equal(train_x[train_x_index], image_1))
        # print(train_y[train_x_index])
        # cv2.imshow('hello', cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # print(train_x_index)
        # print(train_x_index)
        train_x_index += 1
        
test_x = np.zeros((test_count, 128, 128, 3))
test_y = np.zeros((test_count,))
        
file = open(test_label_path, 'r')
Lines = file.readlines()
file.close()

for j in range(1, len(Lines)):
    image = PIL.Image.open(test_path + Lines[j].split(';')[0])
    factor = max(250 / image.size[0], 250 / image.size[1])
    if factor < 1:
        factor = 1
    image = image.resize((int(factor * image.size[0]), int(factor * image.size[1])))
    image_1 = transformer.forward(image)
    image_1 = np.array(image_1)
    test_x[j - 1] = np.copy(image_1)
    test_y[j - 1] = Lines[j].split(';')[7]
    # print(np.array_equal(test_x[j - 1], image_1))
    # print(test_y[j - 1])
    # cv2.imshow('hello', cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # print(j)
    
train_x = train_x / 255
test_x = test_x / 255
train_x = np.swapaxes(train_x,1,3)
train_x = np.swapaxes(train_x,2,3)
test_x = np.swapaxes(test_x,1,3)
test_x = np.swapaxes(test_x,2,3)
train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)
train_y = train_y.astype(np.float32)
test_y = test_y.astype(np.float32)

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

print("train_x shape: ", train_x.size())
print("train_y shape: ", train_y.size())
print("test_x shape: ", test_x.size())
print("test_y shape: ", test_y.size())

print("train_x type: ", train_x.type())
print("train_y type: ", train_y.type())
print("test_x type: ", test_x.type())
print("test_y type: ", test_y.type())

torch.save(train_x, 'GTSRB_train_x.pt')
torch.save(train_y, 'GTSRB_train_y.pt')
torch.save(test_x, 'GTSRB_test_x.pt')
torch.save(test_y, 'GTSRB_test_y.pt')