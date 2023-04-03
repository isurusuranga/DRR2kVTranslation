import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class ImageDataset(Dataset):
    def __init__(self, rootDRR=None, rootkV=None, opt=None, unaligned=True):
        self.rootDRR = rootDRR
        self.rootkV = rootkV
        self.options = opt

        self.drrImages = os.listdir(rootDRR)
        self.kVImages = os.listdir(rootkV)

        self.unaligned = unaligned

        self.drrLen = len(self.drrImages)
        self.kVLen = len(self.kVImages)
        self.lenDataset = max(self.drrLen, self.kVLen)

    def transform(self, image):
        # Resize
        image = image.resize((self.options.img_res, self.options.img_res))
        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5,), (0.5,))

        return image

    def __getitem__(self, index):

        drrImg = self.drrImages[index % self.drrLen]

        if self.unaligned:
            kVImg = self.kVImages[random.randint(0, self.kVLen - 1)]
        else:
            kVImg = self.kVImages[index % self.kVLen]

        drrPath = os.path.join(self.rootDRR, drrImg)
        kVPath = os.path.join(self.rootkV, kVImg)

        drrImg = self.transform(Image.open(drrPath))
        drr_img_name_str = os.path.basename(drrPath).rsplit('.', 1)[0]
        drr_gantry_angle = float(os.path.basename(drr_img_name_str).rsplit('_', 1)[1])
        drr_gantry_angle = torch.tensor([drr_gantry_angle / 360])

        kVImg = self.transform(Image.open(kVPath))
        kv_img_name_str = os.path.basename(kVPath).rsplit('.', 1)[0]
        kv_gantry_angle = float(os.path.basename(kv_img_name_str).rsplit('_', 1)[1])
        kv_gantry_angle = torch.tensor([kv_gantry_angle / 360])

        return {"A": {"img": drrImg, "gantry": drr_gantry_angle}, "B": {"img": kVImg, "gantry": kv_gantry_angle}}

    def __len__(self):

        return self.lenDataset
