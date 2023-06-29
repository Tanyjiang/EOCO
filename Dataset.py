import cv2
from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms import Transforms
import glob
from torchvision.transforms import functional
import pandas as pd
import json

class Dataset(data.Dataset):
    def __init__(self, dataset, increntmal_phase,exemplar_set,exemplar_set_gt):
        self.dataset = dataset
        self.label_list=[]
        self.image_list=[]
        self.target=[]
        train_stage=['part_A_final','jujubes','cherrys','tulips','chickens','vehicles']
        test_stage={
            0: ['part_A_final'],
            1: ['jujubes','part_A_final'],
            2: ['cherrys', 'part_A_final','jujubes'],
            3: ['tulips', 'part_A_final', 'jujubes','cherrys'],
            4: ['chickens', 'part_A_final', 'jujubes', 'cherrys','tulips'],
            5: ['vehicles', 'part_A_final', 'jujubes', 'cherrys', 'tulips', 'chickens'],
        }
        self.class_name=['others','IMG','jujube','cherry','tulip','chicken','vehicle']

        dataset = dataset + '_data'
        if increntmal_phase==0:
            dataset_path = os.path.join('./dataset',train_stage[increntmal_phase],dataset,'images')
            self.image_list = glob.glob(os.path.join(dataset_path, '*.jpg'))
            for index in range(len(self.image_list)):
                image = self.image_list[index]
                image_name=image.split('images/')[1]
                if image_name.startswith('IMG'):
                    self.label_list.append(self.image_list[index].replace('.jpg', '.csv').replace('images', 'ground_truth'))
                    if self.dataset=='train':
                        self.target.append(1)
                else:
                    img = cv2.imread(image)
                    height = img.shape[0]
                    width = img.shape[1]
                    label=np.zeros((height,width))
                    self.label_list.append(label)
                    self.target.append(0)

        elif increntmal_phase>=1:
            if self.dataset=='test' or self.dataset=='val':
                for phase in range(len(test_stage[increntmal_phase])):
                    dataset_path=os.path.join('./dataset',test_stage[increntmal_phase][phase],dataset,'images')
                    img_list_buff=glob.glob(os.path.join(dataset_path,'*.jpg'))
                    for index in range(len(img_list_buff)):
                        self.image_list.append(img_list_buff[index])
                        self.label_list.append(img_list_buff[index].replace('.jpg', '.csv').replace('images', 'ground_truth'))
                        # image_name = img_list_buff[index].split('images/')[1]
                        # image_name = image_name.split('_')[0]
                        # if image_name in self.class_name:
                        #     self.target.append(self.class_name.index(image_name))
                        # else:
                        #     print('error!')

            elif self.dataset=='train':
                dataset_path = os.path.join('./dataset',train_stage[increntmal_phase], dataset, 'images')
                img_list_buff = glob.glob(os.path.join(dataset_path, '*.jpg'))
                for index in range(len(img_list_buff)):
                    self.image_list.append(img_list_buff[index])
                    self.label_list.append(img_list_buff[index].replace('.jpg', '.csv').replace('images', 'ground_truth'))
                    image_name = img_list_buff[index].split('images/')[1]
                    image_name = image_name.split('_')[0]
                    if image_name in self.class_name:
                        self.target.append(self.class_name.index(image_name))
                    else:
                        print('error!')

                if exemplar_set!=None:
                    for index in range(len(exemplar_set)):
                        for num in range(len(exemplar_set[index])):
                            self.image_list.append(exemplar_set[index][num])
                            self.label_list.append(exemplar_set_gt[index][num])
                            image_name = exemplar_set[index][num].split('images/')[1]
                            image_name = image_name.split('_')[0]
                            if image_name in self.class_name:
                                self.target.append(self.class_name.index(image_name))
                            else:
                                self.target.append(0)

    def __getitem__(self, index):
        #class_name = ['others', 'IMG', 'jujube', 'cherry']
        image = Image.open(self.image_list[index]).convert('RGB')
        # target = self.target[index]
        if self.dataset == 'train':
            target = self.target[index]
            img=self.image_list[index].split('images/')[1]
            img = img.split('_')[0]
            #if img.startswith('IMG') or img.startswith('jujube') or img.startswith('cherry'):
            if img in self.class_name:
                label = pd.read_csv((self.label_list[index]), sep=',',header=None).values
            else:
                label = self.label_list[index]
        else:
            label = pd.read_csv((self.label_list[index]), sep=',',header=None).values

        density = np.asarray(label,np.float32)
        attention = np.zeros(density.shape)
        attention[density > 0.0001] = 1
        attention = attention.astype(np.float32, copy=False)
        gt = np.array(np.sum(np.sum(density)))

        trans = Transforms((0.8, 1.2), (400, 400), 1, (0.5, 1.5), self.dataset)
        if self.dataset=='train':
            image, density, attention = trans(image, density,attention)
            return image, density, target, attention

        else:
            height, width = image.size[1], image.size[0]
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image, gt

    def __len__(self):
        return len(self.image_list)


# if __name__ == '__main__':
#     train_dataset = Dataset1('/home/shengqin/wq/incremental_learning/iCaRL-master（modify）/dataset', 'train')
#     train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
#
#     for image, label, att in train_loader:
#         print(image.size())
#         print(label.size())
#         print(att.size())
#
#         img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
#         plt.figure()
#         plt.subplot(1, 3, 1)
#         plt.imshow(img)
#         plt.subplot(1, 3, 2)
#         plt.imshow(label.squeeze(), cmap='jet')
#         plt.subplot(1, 3, 3)
#         plt.imshow(att.squeeze(), cmap='jet')
#         plt.show()
#