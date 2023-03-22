from hydra.utils import instantiate
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image

def create_dataloader(data_cfg, is_training=True):
    transform = instantiate(data_cfg.transform, is_training=is_training)
    print(f'Data augmentation is as follows \n{transform}\n')

    if is_training:
        print("ok",data_cfg.baseinfo.name)
        dataset = instantiate(data_cfg.trainset, transform=transform)
        print(f'{len(dataset)} images and {data_cfg.baseinfo.num_classes} classes were found from {data_cfg.trainset.root}')
    else:
        dataset = instantiate(data_cfg.valset, transform=transform)
        print(f'{len(dataset)} images and {len(dataset.classes)} classes were found from {data_cfg.valset.root}')

    sampler = instantiate(data_cfg.sampler, dataset=dataset, shuffle=is_training)
    dataloader = instantiate(data_cfg.loader, dataset=dataset, sampler=sampler, drop_last=is_training)
    return dataloader


class Random_noise_dataset(Dataset):
    def __init__(self, category_num, image_size, dim=1, transform=None):
        
        self.category_num = category_num
        if dim ==1:
            self.img = np.random.rand(category_num, image_size, image_size)
        if dim ==3:
            self.img = np.random.rand(category_num, image_size, image_size,3)
            
        self.label = torch.range(0, int(category_num-1) )
        self.image_size = image_size
        
        self.len = len(self.label)
        self.transform = transform
        print(self.img.shape)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = self.img[index]
        
        image = Image.fromarray((image*255).astype(np.uint8)).convert("RGB")
        if self.transform:
            
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.astype(np.float32)).clone()
            
        return image, self.label[index]


def create_random_noise_dataloader(data_cfg, is_training=True):
    print("ok1")
    transform = instantiate(data_cfg.transform, is_training=is_training)
    print(f'Data augmentation is as follows \n{transform}\n')

    dataset = Random_noise_dataset(data_cfg.random_noise.category, data_cfg.random_noise.image_size, dim=data_cfg.random_noise.dim, transform=transform)
    
    print(f'{len(dataset)} images')

    sampler = instantiate(data_cfg.sampler, dataset=dataset, shuffle=is_training)
    dataloader = instantiate(data_cfg.loader, dataset=dataset, sampler=sampler, drop_last=is_training)
    return dataloader  