import torch
import os
import glob
import numpy as np
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
normalCrop = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            normalize])

Crop = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor()])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cuda(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cuda(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

class Image_Dataset(Dataset):
    def __init__(self, content_dir, style_dir, max_train_samples, model):
        self.model = model
        content_resize_dir = content_dir + "_input"
        style_resize_dir = style_dir + "_input"
        if not os.path.exists(content_resize_dir):
            os.mkdir(content_resize_dir)
            self.trans(content_dir)
        if not os.path.exists(style_resize_dir):
            os.mkdir(style_resize_dir)
            self.trans(style_dir)
        if max_train_samples != None:
            self.content_images = glob.glob((content_resize_dir + "/*"))[:max_train_samples]
        else:
            self.content_images = glob.glob((content_resize_dir + "/*"))
        self.style_images = glob.glob((style_resize_dir + "/*"))
        np.random.shuffle(self.content_images)
        np.random.shuffle(self.style_images)
#        print("WWW")
#        print(self.content_images)
    
    def __len__(self):
        return len(self.content_images)
        
    def __getitem__(self, index):
        content = Image.open(self.content_images[index])
        style = Image.open(self.style_images[index % len(self.style_images)])
        if self.model == "vgg":
            content = normalCrop(content)
            style = normalCrop(style)
        if self.model == "dense":
            content = Crop(content)
            style = Crop(style)
        return content, style
        
    def trans(self, image_dir):
        #print(image_dir)
        image_input_dir = image_dir + "_input"
        files = os.listdir(image_dir)
        #print(files)
        for i in files:
            filename = os.path.basename(i)
            #print(filename)
            image = io.imread(os.path.join(image_dir, i))
            if len(image.shape) != 3 and image.shape[-1] != 3:
                continue
            #print(image.shape)
            H, W, _ = image.shape
            if H > W:
                H = int(H/W*512)
                W = 512
            else:
                W = int(W/H*512)
                H = 512
            image = transform.resize(image, (H, W), mode="reflect", anti_aliasing=True)
            #print(os.path.join(image_input_dir, filename))
            io.imsave(os.path.join(image_input_dir, filename), image)
        
