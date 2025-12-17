import numpy as np
import torch
from torch.utils.data import Dataset
import sys

################################################################################################################################################
################################################################################################################################################

class ClassificationDataset(Dataset):
    """
    Dataset for training the window classification model.

    input: multi-variable time-series sequences
    labels: track not contaminated/track contaminated/shower label
    """
    def __init__(self, device, input, labels):
        self.device = device
        self.input = torch.as_tensor(input, dtype=torch.float)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx].to(self.device), self.labels[idx].to(self.device)

################################################################################################################################################
################################################################################################################################################

def get_classification_datasets(device, training_fraction):

    inputs = np.load(sys.path[0] + '/files/x_tokens_UVW.npy')
    labels = np.load(sys.path[0] + '/files/is_contaminated_UVW.npy')  
    
    split_idx = int(labels.shape[0] * training_fraction)
    shuffled_indices = np.random.permutation(labels.shape[0])

    training_dataset = ClassificationDataset(device, 
                                             inputs[shuffled_indices[:split_idx]], 
                                             labels[shuffled_indices[:split_idx]])
    
    validation_dataset = ClassificationDataset(device, 
                                               inputs[shuffled_indices[split_idx:]], 
                                               labels[shuffled_indices[split_idx:]])

    return training_dataset, validation_dataset

################################################################################################################################################
################################################################################################################################################

class SplitPointDataset(Dataset):
    """
    Dataset for training the split point model.

    input: multi-variable time-series sequences
    labels: ground-truth label for each sequence element
    is_contaminated: whether the window is contaminated
    """
    def __init__(self, device, input, labels, is_contaminated):
        self.device = device
        self.input = torch.as_tensor(input, dtype=torch.float)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.is_contaminated = torch.as_tensor(is_contaminated, dtype=torch.long)      

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx].to(self.device), self.labels[idx].to(self.device), self.is_contaminated[idx].to(self.device)

################################################################################################################################################
################################################################################################################################################

def get_split_point_datasets(device, training_fraction):

    inputs = np.load(sys.path[0] + '/files/x_tokens_UVW.npy')
    labels = np.load(sys.path[0] + '/files/y_tokens_UVW.npy')
    is_contaminated = np.load(sys.path[0] + '/files/is_contaminated_UVW.npy')  
   
    split_idx = int(labels.shape[0] * training_fraction)
    shuffled_indices = np.random.permutation(labels.shape[0])

    training_dataset = SplitPointDataset(device, 
                                         inputs[shuffled_indices[:split_idx]], 
                                         labels[shuffled_indices[:split_idx]],  
                                         is_contaminated[shuffled_indices[:split_idx]])
    
    validation_dataset = SplitPointDataset(device, 
                                           inputs[shuffled_indices[split_idx:]], 
                                           labels[shuffled_indices[split_idx:]],
                                           is_contaminated[shuffled_indices[split_idx:]])

    return training_dataset, validation_dataset