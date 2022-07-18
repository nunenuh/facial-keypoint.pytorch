import glob
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import pandas as pd


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.key_pts_frame)
    
    def _load_image_path(self, idx):
        return os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
    
    def _load_image(self, impath):
        image = mpimg.imread(impath)
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
            
        return image
    
    def _load_pts(self, idx):
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        return key_pts
    
    def __getitem__(self, idx):
        impath = self._load_image_path(idx)
        image = self._load_image(impath)
        key_pts = self._load_pts(idx)
        
        sample = {'image': image, 'keypoints': key_pts}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample