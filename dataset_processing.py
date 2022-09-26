import torch
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data.dataset import Dataset

import numpy as np
new_class = {0, 0, 0, 0, 0, 0}
new_class_list = ['Hat', 'Sunglasses', 'Mask', 'Helmet', 'Call', 'Negative']
class DatasetProcessing(Dataset):

    def __init__(self, data_root, img_list, transform=None, aug_transfrom=None):
        self.img_path = list()
        label = list()

        self.img_list = img_list
        self.transform = transform
        self.aug_transfrom = aug_transfrom
        fp = open(img_list, 'r', encoding='utf-8')
        while(True):
            line = fp.readline().strip('\n').strip(',')
            if not line:
                break
            elem = line.split('\t')
            elem[0] = data_root + elem[0]
            self.img_path.append(elem[0])
            # label.append(list(map(int, elem[1].split(' '))))
            if elem[1].split(' ')[-1] == '':
                label.append(list(map(int, elem[1].split(' ')[:-1])))
            else:
                label.append(list(map(int, elem[1].split(' '))))

        self.label = np.asarray(label, dtype=float)
        label_sum = np.sum(self.label, axis = 0)
        label_average = np.average(self.label, axis = 0)
        # average of each class will be used to pos_ratio in BCE Loss
        self.pos_ratio = np.asarray(label_average, dtype=float)
        print("\tlabel count = ", label_sum,", ",  "label dim = ", np.shape(self.label))
        print("\tpositive ratio = ", label_average)
        fp.close()
        self.label = np.asarray(label, dtype=np.float32)

    def label_count(self, labels):
        for label_ in labels:
            for i in range(0, len(label_)):
                if label_[i] == 1:
                    new_class[i] += 1
                else:
                    pass
        return new_class

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img)
        if self.aug_transfrom is not None:
            img = self.aug_transfrom(image=img)["image"]
        #print(self.label)
        label = torch.from_numpy(self.label[index])
        return (img, label)

    def __len__(self):
        return len(self.img_path)


"""
    no use txt files
"""
class onlyImageDataset(Dataset):

    def __init__(self, img_list_folder, transform=None):
        self.img_path = img_list_folder
        self.file_names = os.listdir(img_list_folder)
        self.transform = transform


    def __getitem__(self, index):
        RGB_trans = transforms.ToTensor()
        img = Image.open(self.img_path + self.file_names[index])
        img = img.convert('RGB')
        img_RGB = img.copy()
        img_RGB = img_RGB.resize((224, 224))
        img_RGB = RGB_trans(img_RGB)
        if self.transform is not None:
            img = self.transform(img)
        return (img, img_RGB)

    def __len__(self):
        return len(self.img_path)