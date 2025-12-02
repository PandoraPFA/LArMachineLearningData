import numpy as np
import torch
from torch.utils.data import Dataset

################################################################################################################################################
################################################################################################################################################

class ClassificationDataset(Dataset):
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
 
    inputs = np.load('x_tokens_UVW.npy', mmap_mode='r')
    labels = np.load('is_contaminated_UVW.npy', mmap_mode='r')
    
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
    
    inputs = np.load('x_tokens_UVW.npy', mmap_mode='r')
    labels = np.load('y_tokens_UVW.npy', mmap_mode='r')
    is_contaminated = np.load('is_contaminated_UVW.npy', mmap_mode='r')
   
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