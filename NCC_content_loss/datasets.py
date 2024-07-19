import torch
from torch.utils.data import Dataset

class fMRIDataset(Dataset):

    def __init__(self, data_src, layer_features, layer_name):
        self.data_src = data_src
        self.data_tar = layer_features[layer_name]

    def __getitem__(self, index):
        item_src = self.data_src[index]
        item_tar = self.data_tar[index]
        return {'A': item_src, 'B': item_tar}

    def __len__(self):
        return max(len(self.data_src), len(self.data_tar))