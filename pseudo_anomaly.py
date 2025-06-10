import cv2
import noise
import torch
import random
import numpy as np
from PIL import Image
import albumentations as A
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import ImageFilter
import io


bounds = {
    "mvtec" : {
        "toothbrush":(14, 39), "cable":(14, 65), "screw":(9, 35), "transistor":(14, 86), "capsule":(14, 39) , "bottle":(14, 65), "hazelnut":(14, 65), "metal_nut":(14, 65), "pill":(14, 39), "zipper":(14, 65),    
        "wood":(14, 86), "carpet":(14, 86), "grid":(14, 86), "leather":(14, 86), "tile":(14, 86)
    },
    'visa' : {
        'candle':(14, 86), 'capsules':(6, 86), 'cashew':(10, 86), 'chewinggum':(14, 86), 'fryum':(6, 15), 'macaroni1':(14, 86),
        'macaroni2':(14, 86), 'pcb1':(14, 86), 'pcb2':(14, 86), 'pcb3':(14, 86), 'pcb4':(14, 86), 'pipe_fryum':(14, 86)
    },
    'mpdd' : {
        'bracket_black':(6, 86), 'bracket_brown':(6, 86), 'bracket_white':(6, 86), 'connector':(6, 86), 'metal_plate':(14, 86), 'tubes':(6, 86) 
    },
    'btad' : {
        '01':(9, 86), '02':(14, 86), '03':(9, 86)
    },
    'dtd' : {
        'Blotchy_099':(14, 86) , 'Fibrous_183':(14, 86) , 'Marbled_078':(14, 86) , 'Matted_069':(14, 86) , 'Mesh_114':(14, 86) , 'Perforated_037':(14, 86) , 'Stratified_154':(14, 86) , 'Woven_001':(14, 86) , 'Woven_068':(14, 86) , 'Woven_104':(14, 86) , 'Woven_125':(14, 86) , 'Woven_127':(14, 86)
    },
    'brats2021' : {
        "":(14, 86)
    },
    'headct' : {
        "":(9, 86)
    },
    'wfdd' : {
        "grey_cloth":(9, 86), "grid_cloth":(9, 86), "pink_flower":(9, 86), "yellow_cloth":(9, 86)
    }
}

class RandomAugmentations:
    def __init__(self, seed=None):
        self.seed = seed

        self.param_ranges = {
            'brightness': {'light': (0.1, 0.1), 'medium': (0.4, 0.4), 'heavy': (0.8, 1)},
            'contrast':   {'light': (0.1, 0.1), 'medium': (0.4, 0.4),  'heavy': (0.8, 1)},
            'saturation': {'light': (0.1, 0.1), 'medium': (0.4, 0.4), 'heavy': (0.8, 1)},
            'hue':        {'light': (0.1, 0.1), 'medium': (0.3, 0.3), 'heavy': (0.5, 0.5)},

            'elastic_alpha': {'light': (10, 20), 'medium': (20, 40), 'heavy': (40, 100)},

            'torn_lines': {'light': (1, 3), 'medium': (5, 10), 'heavy': (10, 20)},
            
            'perlin_scale': {'light': (20, 50), 'medium': (10, 20), 'heavy': (5, 10)},
            'perlin_threshold': {'light': (200, 255), 'medium': (150, 200), 'heavy': (128, 150)},

            'swirl_strength': {'light': (0.5, 1.0), 'medium': (1.0, 1.5), 'heavy': (1.5, 2)},

            'erase_ratio': {'light': (0.01, 0.05), 'medium': (0.05, 0.1), 'heavy': (0.1, 0.2)},
            'erase_rects': {'light': (1, 2), 'medium': (2, 3), 'heavy': (3, 5)},

            'blur_radius': {'light': (0.2, 0.5), 'medium': (0.8, 1.5), 'heavy': (2.0, 3)},

            'jpeg_quality': {'light': (30, 50), 'medium': (20, 30), 'heavy': (1, 20)},
        }

        self.num_augmentations = {'light': 1, 'medium': 2, 'heavy': 4}

        self.augmentations = {
            "light": [
                self.gaussian_blur, self.elastic_transform, self.swirl_distortion,
            ],
            "medium": [
                self.gaussian_blur, self.swirl_distortion, self.jpeg_artifacts, 
                self.elastic_transform
            ],
            "heavy": [
                self.elastic_transform, self.torn_paper_effect, self.jpeg_artifacts,
                self.perlin_noise_mask, self.swirl_distortion, self.random_erasing, 
                self.gaussian_blur,
            ]
        }


    def apply(self, image, level='medium'):
        image_np = np.array(image)
        n_augmentations = random.randint(0, self.num_augmentations[level])
        selected_augmentations = random.sample(self.augmentations[level], n_augmentations)
        
        selected_augmentations.insert(random.randint(0, len(selected_augmentations)), self.color_transformation)

        for augmentation in selected_augmentations:
            image_np = augmentation(image_np, level)

        return Image.fromarray(image_np)

    def elastic_transform(self, image, level):
        alpha = random.uniform(*self.param_ranges['elastic_alpha'][level])
        sigma = 3.0
        random_state = np.random.RandomState(self.seed)
        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="reflect") * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="reflect") * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).flatten(), (x + dx).flatten()

        distorted_image = np.zeros_like(image)
        for i in range(shape[2]):
            distorted_image[..., i] = map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(shape[:2])
        return distorted_image

    def torn_paper_effect(self, image, level):
        image_np = image.copy()
        height, width = image_np.shape[:2]
        num_lines = random.randint(*self.param_ranges['torn_lines'][level])
        for _ in range(num_lines):
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            end_x = np.random.randint(0, width)
            end_y = np.random.randint(0, height)
            cv2.line(image_np, (start_x, start_y), (end_x, end_y), [random.choice([0, 255]) for _ in range(3)], thickness=1)
        return image_np

    def perlin_noise_mask(self, image, level):
        scale = random.uniform(*self.param_ranges['perlin_scale'][level])
        threshold = random.randint(*self.param_ranges['perlin_threshold'][level])
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                mask[i, j] = noise.pnoise2(i / scale, j / scale, octaves=6)
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
        image[mask > threshold] = np.random.randint(0, 255, 3)
        return image

    def color_transformation(self, image, level):
        b = random.uniform(*self.param_ranges['brightness'][level])
        c = random.uniform(*self.param_ranges['contrast'][level])
        s = random.uniform(*self.param_ranges['saturation'][level])
        h = random.uniform(*self.param_ranges['hue'][level])
        transform = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
        return np.array(transform(Image.fromarray(image)))

    def swirl_distortion(self, image, level):
        strength = random.uniform(*self.param_ranges['swirl_strength'][level])
        patch_np = np.array(image)
        height, width = patch_np.shape[:2]
        center_x, center_y = width // 2, height // 2
        y, x = np.indices((height, width))
        x = x - center_x
        y = y - center_y
        distance = np.sqrt(x**2 + y**2)
        angle = strength * np.exp(-distance**2 / (2 * (min(height, width) // 3)**2))
        new_x = center_x + x * np.cos(angle) - y * np.sin(angle)
        new_y = center_y + x * np.sin(angle) + y * np.cos(angle)
        map_x = np.clip(new_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, height - 1).astype(np.float32)
        return cv2.remap(patch_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def random_erasing(self, image, level):
        image_np = image.copy()
        h, w = image_np.shape[:2]
        erase_area_ratio = random.uniform(*self.param_ranges['erase_ratio'][level])
        num_rectangles = random.randint(*self.param_ranges['erase_rects'][level])
        for _ in range(num_rectangles):
            erase_area = int(erase_area_ratio * h * w)
            erase_aspect_ratio = random.uniform(0.3, 3.3)
            erase_height = int(np.sqrt(erase_area / erase_aspect_ratio))
            erase_width = int(erase_aspect_ratio * erase_height)
            x = random.randint(0, max(0, w - erase_width))
            y = random.randint(0, max(0, h - erase_height))
            image_np[y:y+erase_height, x:x+erase_width] = np.random.randint(0, 255, 3)
        return image_np

    def gaussian_blur(self, image, level):
        radius = random.uniform(*self.param_ranges['blur_radius'][level])
        return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius)))

    def jpeg_artifacts(self, image, level):
        quality = random.randint(*self.param_ranges['jpeg_quality'][level])
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        return np.array(Image.open(buffer))
    

class AnomalyGenerator(object):
    def __init__(self, dataset, class_name, seed):
        self.lower_bound, self.upper_bound = bounds[dataset][class_name]

        self.random_augmentor = RandomAugmentations(seed=seed)

    def rotate(self, patch, width, height, min_angle=-90, max_angle=90):
        random_rotate = random.uniform(min_angle, max_angle)
        patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
        patch = patch.resize((width, height), resample=Image.BICUBIC)
        mask = patch.split()[-1]
        
        return patch.convert("RGB"), mask

    def intersect_masks(self, mask1, mask2):
        mask1_np = np.array(mask1)
        mask2_np = np.array(mask2)
    
        intersection = np.logical_and(mask1_np, mask2_np).astype(np.uint8) * 255
        intersection_mask = Image.fromarray(intersection)
    
        return intersection_mask
    
    def get_max_shape(self, x, y, foreground_mask):
        max_width = 0
        for i in range(x, foreground_mask.shape[1]):
            if foreground_mask[y, i] == 1:
                max_width += 1
            else:
                break

        max_height = 0
        for j in range(y, foreground_mask.shape[0]):
            if foreground_mask[j, x] == 1:
                max_height += 1
            else:
                break
                
        return max_width, max_height
    
    def sample_patch_size(self, foreground_mask, x1, y1, x2, y2, max_width, max_height):
        num_attempts = 0
        while num_attempts <= 10:
            patch_width = random.randint(0, max_width)
            patch_height = random.randint(0, max_height)

            patch_region_src = foreground_mask[y1:y1+patch_height, x1:x1+patch_width]
            patch_region_dst = foreground_mask[y2:y2+patch_height, x2:x2+patch_width]

            if np.all(patch_region_src == 1) and np.all(patch_region_dst == 1):
                break
                
            num_attempts += 1

        return patch_width, patch_height
    
    def expand_mask(self, mask, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=10)
        
        return expanded_mask
    
    def sample_coordinate_shape(self, foreground_mask):
        foreground_mask = self.expand_mask(foreground_mask)
        
        h, w = foreground_mask.shape
        coords = np.column_stack(np.where(foreground_mask == 1))
        
        num_attempts = 0
        while num_attempts <= 250:
            y1, x1 = coords[random.randint(0, len(coords) - 1)]
            max_width1, max_height1 = self.get_max_shape(x1, y1, foreground_mask)       

            y2, x2 = coords[random.randint(0, len(coords) - 1)]
            max_width2, max_height2 = self.get_max_shape(x2, y2, foreground_mask) 

            max_width, max_height = min(max_width1, max_width2), min(max_height1, max_height2)

            patch_width, patch_height = self.sample_patch_size(foreground_mask, x1, y1, x2, y2, max_width, max_height)

            if patch_width < self.lower_bound or patch_height < self.lower_bound:
                num_attempts += 1

            else:
                patch_region_src = foreground_mask[y1:y1+patch_height, x1:x1+patch_width]
                patch_region_dst = foreground_mask[y2:y2+patch_height, x2:x2+patch_width]

                if np.all(patch_region_src == 1) and np.all(patch_region_dst == 1) and (x2 + patch_width <= w and y2 + patch_height <= h):
                    break

                num_attempts += 1
        
        if num_attempts > 250:
            y1, x1 = coords[random.randint(0, len(coords) - 1)]
            y2, x2 = coords[random.randint(0, len(coords) - 1)]
            patch_width, patch_height = self.lower_bound, self.lower_bound

        return x1, y1, x2, y2, patch_width, patch_height
            
    def __call__(self, imgs, foreground_masks):
        batch_size, _, h, w = imgs.shape
        transformed_imgs = []
        transformed_masks = []

        for i in range(batch_size):
            img = imgs[i].cpu()
            img_pil = transforms.ToPILImage()(img)
            foreground_mask = foreground_masks[i].cpu().squeeze(0).numpy()

            x1, y1, x2, y2, patch_width, patch_height = self.sample_coordinate_shape(foreground_mask)

            patch = img_pil.crop((x1, y1, x1 + int(patch_width), y1 + int(patch_height)))

            # transformations 
            patch = self.random_augmentor.apply(patch, np.random.choice(['light', 'medium', 'heavy'], p=[0.2,0.2,0.6]))

            patch, rotation_mask = self.rotate(patch, patch_width, patch_height)
            
            mask = np.ones((int(patch_height), int(patch_width)), dtype=np.uint8)
            mask = cv2.resize(mask, (int(patch_width), int(patch_height)), interpolation=cv2.INTER_CUBIC)
            mask = self.intersect_masks(mask, rotation_mask)
                                    
            augmented = img_pil.copy()

            augmented.paste(patch, (x2, y2), mask=mask)

            org_mask = Image.fromarray(np.zeros((h, w), dtype='uint8')).convert('L')
            org_mask.paste(mask, (x2, y2))
            
            augmented = transforms.ToTensor()(augmented)
            org_mask = transforms.ToTensor()(org_mask)

            transformed_imgs.append(augmented)
            transformed_masks.append(org_mask)

        transformed_imgs = torch.stack(transformed_imgs)
        transformed_masks = torch.stack(transformed_masks)

        return transformed_imgs, transformed_masks