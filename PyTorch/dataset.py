import torchvision
from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(torchvision.datasets.VOCDetection):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
    
    def __getitem__(self, idx):
        image = Image.open()
    
    def __len__(self):
        return len()

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class YOLODataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
    
    def __getitem__(self, idx):
        image = Image.open()
    
    def __len__(self):
        return len()