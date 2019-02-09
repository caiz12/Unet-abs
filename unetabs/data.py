""" Organize data  """

import numpy as np

from torch.utils.data import Dataset


class AbsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, flux_file, lbl_file, mode='train', transform=None, testing=False):
        """
        Args:
            flux_file (string): Path to the csv file with annotations.
            lbl_file (string):
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        all_flux = np.load(flux_file)
        all_lbls = np.load(lbl_file)
        # Train, validate or test?
        if mode == 'train':
            if testing:
                self.all_flux = all_flux[:,0:1000]
                self.all_lbls = all_lbls[:,0:1000]
            else:
                self.all_flux = all_flux[:,0:9500]
                self.all_lbls = all_lbls[:,0:9500]
        elif mode == 'validate':
            self.all_flux = all_flux[:,-500:]
            self.all_lbls = all_lbls[:,-500:]

        self.transform = transform

    def __len__(self):
        return self.all_flux.shape[1]

    def __getitem__(self, idx):
        spec = self.all_flux[:, idx]
        isz = int(np.sqrt(spec.size))
        image = spec.reshape((1,isz,isz)).astype(np.float32)
        # Labels
        lbls = self.all_lbls[:,idx]
        lbls_image = lbls.reshape((1,isz,isz)).astype(np.float32)
        sample = {'image': image, 'labels': lbls_image}

        #if self.transform:
        #    sample = self.transform(sample)

        return sample