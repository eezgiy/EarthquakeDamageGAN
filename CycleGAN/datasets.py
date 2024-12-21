import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Gri tonlamalı görüntüyü RGB'ye dönüştüren fonksiyon
def check_and_convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Gri tonlamalı ise RGB'ye dönüştür
    return image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.png'))

    def __getitem__(self, index):
        # Dosyayı aç ve RGB'ye dönüştür
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_A = check_and_convert_to_rgb(image_A)
        item_A = self.transform(image_A)

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            image_B = check_and_convert_to_rgb(image_B)
            item_B = self.transform(image_B)
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
            image_B = check_and_convert_to_rgb(image_B)
            item_B = self.transform(image_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
