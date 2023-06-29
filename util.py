import glob
import json
import dataset
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from torch.autograd import Variable
from Dataset import Dataset
import os
import cv2
import time
import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class EoCNet:

    def __init__(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate):

        super(EoCNet, self).__init__()
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(feature_extractor)
        self.exemplar_set = []
        self.exemplar_set_gt=[]
        self.class_mean_set = []
        self.numclass = numclass
        self.increntmal_phase= 0
        self.train_list=list()
        self.val_list = list()
        self.batchsize = batch_size
        self.memory_size=memory_size
        self.task_size=task_size
        self.workers=4
        self.train_loader=None
        self.test_loader=None
        self.train_dataset=[]
        self.image_list = list()
        self.label_list = list()
        self.class_name = ['others', 'IMG', 'jujube', 'cherry', 'tulip', 'chicken', 'vehicle']
        self.transform = transforms.Compose([
            transforms.Resize([400,400]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _test(self, test_loader,flag,model):
        model.eval()
        mae_final = 0
        mse_final = 0
        # accuracy_final =0
        mae = 0
        mae_total = 0
        mse_total = 0
        class_num=0
        # correct=0
        if flag == 'test':
            num = {0: [0, 182], 1: [0, 248, 430], 2: [0, 182, 364, 612], 3: [0, 215, 397, 645, 827],
                   4: [0, 182, 364, 612, 794, 1009], 5: [0, 176, 358, 606, 788, 1003, 1185]
                   }
        elif flag == 'val':
            num = {0: [0, 59], 1: [0, 50, 109], 2: [0, 50, 109, 159], 3: [0, 40, 99, 149, 199],
                   4: [0, 40, 99, 149, 199, 239], 5: [0, 44, 103, 153, 203, 243, 283]
                   }
        bottom=0
        upper=1
        all_test_dataset_img=test_loader.dataset.image_list
        all_test_dataset_label=test_loader.dataset.label_list
        all_test_dataset_target=test_loader.dataset.target
        for index_ in range(len(num[self.increntmal_phase])-1):
            #correct = 0
            mae = 0
            mse = 0
            test_loader.dataset.image_list = all_test_dataset_img[num[self.increntmal_phase][bottom]:num[self.increntmal_phase][upper]]
            test_loader.dataset.label_list = all_test_dataset_label[num[self.increntmal_phase][bottom]:num[self.increntmal_phase][upper]]
            #test_loader.dataset.target = all_test_dataset_target[num[self.increntmal_phase][bottom]:num[self.increntmal_phase][upper]]
            for i, (img, density) in enumerate(test_loader):
                model=model.to(device)
                density=density.to(device)
                # target = target.to(device)
                img=img.to(device)
                with torch.no_grad():
                    output, cls, _, _ = model(img, 1)

                for index in range(cls.shape[0]):
                    channel_num = torch.argmax(cls[index])
                    output = output[:, channel_num:channel_num + 1, :, :]
                    mae += abs(output.data.sum() - density.sum().type(torch.FloatTensor).cuda())
                    mse += ((output.data.sum()-density.sum()) ** 2).item()
                    mae_total += abs(output.data.sum() - density.sum().type(torch.FloatTensor).cuda())
                    mse_total += ((output.data.sum() - density.sum()) ** 2).item()

            #accuracy = correct/len(test_loader.dataset.image_list)
            mae = mae/len(test_loader.dataset.image_list)
            mse = mse/len(test_loader.dataset.image_list)
            mse = mse ** 0.5
            print('class:%d,mae:%.2f，mse:%.2f' % (class_num,mae,mse))
            #accuracy_final += accuracy
            mae_final += mae
            mse_final += mse
            bottom = upper
            upper = upper + 1
            class_num+=1
        #accuracy_final= accuracy_final/(self.increntmal_phase + 1)
        mae_final = mae_final / (self.increntmal_phase + 1)
        mse_final = mse_final / (self.increntmal_phase + 1)
        test_loader.dataset.image_list = all_test_dataset_img
        test_loader.dataset.label_list = all_test_dataset_label
        test_loader.dataset.target = all_test_dataset_target
        print(' * Average MAE :%.2f ' % (mae_final))
        print(' * Average MSE :%.2f ' % (mse_final))

        mae_total = mae_total / len(all_test_dataset_img)
        mse_total = mse_total / len(all_test_dataset_img)
        mse_total = mse_total ** 0.5
        print(' ** MAE :%.2f ' % (mae_total))
        print(' ** MSE :%.2f ' % (mse_total))
        return mae_final

    def test(self):
        print('*begin test*')
        if self.numclass>self.task_size:
           self.model.Incremental_learning_weight(self.numclass)

        checkpoint_best = torch.load(os.path.join('./checkpoint', 'checkpoint_best_'+str(self.increntmal_phase)+'.pth'))
        self.model.load_state_dict(checkpoint_best['model'])
        self.numclass += self.task_size

        test_dataset = Dataset('test', self.increntmal_phase, None, None)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print('{0}th phase:the length of the test_dataset:{1}'.format(self.increntmal_phase, len(test_dataset)))
        mae=self._test(test_loader,'test',self.model)
        self.increntmal_phase += 1

    def transform_image(self, image):
        image = Image.open(image).convert('RGB')
        height, width = image.size[1], image.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        if height > 2000 or width > 2000:
            height,width = 2000, 2000
            image = image.resize((width, height), Image.BILINEAR)
        else:
            image = image.resize((width, height), Image.BILINEAR)
        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image

