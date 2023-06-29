from PIL import Image
import numpy as np
import cv2
import random
from torchvision.transforms import functional


class Transforms(object):
    def __init__(self, scale, crop, stride, gamma, dataset):
        self.scale = scale  #(0.8,1.2)
        self.crop = crop    #(400,400)
        self.stride = stride #1
        self.gamma = gamma   #(0.5,1.5)
        self.dataset = dataset   # 'train'

    def __call__(self, image, density, attention):
        # random resize
        height, width = image.size[1], image.size[0]
        if self.dataset == 'train':
            if height < width:
                short = height
            else:
                short = width
            if short < 512:
                scale = 512 / short
                height = round(height * scale)   #返回浮点数x的四舍五入值。
                width = round(width * scale)
                image = image.resize((width, height), Image.BILINEAR)
                density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
                attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        scale = random.uniform(self.scale[0], self.scale[1])  #random.uniform(参数1，参数2) 返回参数1和参数2之间的任意值
        height = round(height * scale)
        width = round(width * scale)
        image = image.resize((width, height), Image.BILINEAR)
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
        attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        # random crop
        h, w = self.crop[0], self.crop[1]
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]
        attention = attention[dh:dh + h, dw:dw + w]

        # random flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT) #图像的翻转
            density = density[:, ::-1]
            attention = attention[:, ::-1]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)  #对一张图片进行gamma校正,返回：gamma校正的图片

        # random to gray
        if self.dataset == 'train':
            if random.random() < 0.1:
                image = functional.to_grayscale(image, num_output_channels=3)   #作用：将图像转换为灰度图像

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        density = cv2.resize(density, (density.shape[1] // self.stride, density.shape[0] // self.stride),
                             interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        attention = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride),
                               interpolation=cv2.INTER_LINEAR)
        
        attention[attention > 0.0001] = 1
        attention = attention.astype(np.float32, copy=False)

        density = np.reshape(density, [1, density.shape[0], density.shape[1]])
        attention = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])

        return image, density, attention
