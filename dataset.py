import os
import sys
import torch
import random
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class MVTec(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size
        self.has_fg_mask = class_name in ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'zipper']

        path = os.path.join(path, class_name)
        mv_str = '_mask.'

        normal_dir = os.path.join(path, split, "good")
        
        if split == 'train' and self.has_fg_mask:
            self.foreground_mask_path = os.path.join(path, split, "foreground_mask")
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            foreground_mask_dir = None
            if split == 'train' and self.has_fg_mask:
                foreground_mask_dir = os.path.join(self.foreground_mask_path, img_file)
            self.data.append((image_dir, None, foreground_mask_dir))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask
    

class MPDD(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size
        self.has_fg_mask = class_name in ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

        path = os.path.join(path, class_name)
        mv_str = '_mask.'

        normal_dir = os.path.join(path, split, "good")
        
        if split == 'train' and self.has_fg_mask:
            self.foreground_mask_path = os.path.join(path, split, "foreground_mask")
            os.makedirs(self.foreground_mask_path, exist_ok=True)
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            foreground_mask_dir = None
            if split == 'train' and self.has_fg_mask:
                foreground_mask_dir = os.path.join(self.foreground_mask_path, img_file)
            self.data.append((image_dir, None, foreground_mask_dir))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask
    
class BTAD(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size
        self.has_fg_mask = class_name in ['01', '03']

        path = os.path.join(path, class_name)
        
        normal_dir = os.path.join(path, split, "ok")
        
        if split == 'train' and self.has_fg_mask:
            self.foreground_mask_path = os.path.join(path, split, "foreground_mask")
            os.makedirs(self.foreground_mask_path, exist_ok=True)
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            foreground_mask_dir = None
            if split == 'train' and self.has_fg_mask:
                foreground_mask_dir = os.path.join(self.foreground_mask_path, img_file)
            self.data.append((image_dir, None, foreground_mask_dir))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    if class_name in ["01", "02"]:
                        mask_dir = mask_dir.replace('.bmp', '.png')
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask

class VisA(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.path_normal = os.path.join(path, class_name, "Data", "Images", "Normal")
        self.path_anomaly = os.path.join(path, class_name, "Data", "Images", "Anomaly")
        self.foreground_mask_path = os.path.join(path, class_name, "Data", "Images", "foreground_mask")
        self.normal_test = []

        self.class_name = class_name
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size
        img_count = 0

        for filename in os.listdir(self.path_normal):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                img_count += 1
                                
        for img_path in os.listdir(self.path_normal):
            foreground_mask_dir = os.path.join(self.foreground_mask_path, img_path)
            if not os.path.exists(foreground_mask_dir):
                self.normal_test.append(img_path)
                img_count -= 1

        if split == 'train':
            for img_path in os.listdir(self.path_normal):
                image_dir = os.path.join(self.path_normal, img_path)
                if img_path not in self.normal_test:
                    foreground_mask_dir = os.path.join(self.foreground_mask_path, img_path)
                    self.data.append((image_dir, None, foreground_mask_dir))

        elif split == 'test':
            for img_path in self.normal_test:
                self.data.append((os.path.join(self.path_normal, img_path), None, None)) 

            for img_path in os.listdir(self.path_anomaly):
                image_dir = os.path.join(self.path_anomaly, img_path)
                mask_dir = image_dir.replace("Images", "Masks")[:-3] + "png"
                self.data.append((image_dir, mask_dir, None)) 
                
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask
    
class DTD(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size

        path = os.path.join(path, class_name)
        mv_str = '_mask.'

        normal_dir = os.path.join(path, split, "good")
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            self.data.append((image_dir, None, None))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask
    
class BraTS2021(data.Dataset):
    def __init__(self, path, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size

        normal_dir = os.path.join(path, split, "normal")
        
        if split == 'train':
            self.foreground_mask_path = os.path.join(path, split, "foreground_mask")
            os.makedirs(self.foreground_mask_path, exist_ok=True)
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            foreground_mask_dir = None
            if split == 'train':
                foreground_mask_dir = os.path.join(self.foreground_mask_path, img_file)
            self.data.append((image_dir, None, foreground_mask_dir))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = os.path.join(path, "ground_truth", img_file)
                    mask_dir = mask_dir.replace("flair", "seg")
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask
    
class HeadCT(data.Dataset):
    def __init__(self, path, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size

        mv_str = '_mask.'

        normal_dir = os.path.join(path, split, "good")
        
        if split == 'train':
            self.foreground_mask_path = os.path.join(path, split, "foreground_mask")
            os.makedirs(self.foreground_mask_path, exist_ok=True)
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            foreground_mask_dir = None
            if split == 'train':
                foreground_mask_dir = os.path.join(self.foreground_mask_path, img_file)
            self.data.append((image_dir, None, foreground_mask_dir))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask
    
class WFDD(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train', size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self.size = size

        path = os.path.join(path, class_name)
        mv_str = '_mask.'

        normal_dir = os.path.join(path, split, "good")
            
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            self.data.append((image_dir, None, None))
            
        if split == 'test':
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir, None))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, fore_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        C, W, H = image.shape
        foreground_mask = torch.ones((H, W))
        if fore_path:
            foreground_mask = Image.open(fore_path).convert('L')
            foreground_mask = foreground_mask.resize((self.size, self.size), Image.NEAREST)
            foreground_mask = transforms.ToTensor()(foreground_mask)
            
        return image, label, mask, foreground_mask