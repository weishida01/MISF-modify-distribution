import os
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from my_utils.base_dataset import obj_img




class Dataset(torch.utils.data.Dataset):
    def __init__(self, config,mode="train"):
        super(Dataset, self).__init__()
        self.mode = mode
        self.root_path = config.root_path

        if self.mode == 'train':
            path = os.path.join(self.root_path, 'train')
            self.paths = sorted(self.make_dataset(path))
            self.size = len(self.paths)  # get the size of dataset A

        elif self.mode == 'val':
            path = os.path.join(self.root_path, 'val')
            self.paths = sorted(self.make_dataset(path))
            self.size = len(self.paths[:5])  # get the size of dataset A

        elif self.mode == 'test':
            path = os.path.join(self.root_path, 'test')
            self.paths = sorted(self.make_dataset(path))
            self.size = len(self.paths[:5])  # get the size of dataset A

        print('mode:{}'.format(mode))


    def __len__(self):
        return self.size


    def make_dataset(self,dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images


    def __getitem__(self, index):

        path = self.paths[index % self.size]  # make sure index is within then range

        img_in, img_gt = obj_img(path, self.root_path)

        img_in = F.to_tensor(img_in).float()
        img_gt = F.to_tensor(img_gt).float()

        return img_in, img_gt, path



