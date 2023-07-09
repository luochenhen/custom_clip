import torch
from PIL import Image
from torch.utils.data import Dataset

class coco_crop_dataset(Dataset):

    def __init__(self, image_path, image_class, transform=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.image_class = image_class
        self.transform = transform
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img = Image.open(self.image_path[index])
        label = self.image_class[index]
        if self.transform:
            img = self.transform(img)
        return img, label