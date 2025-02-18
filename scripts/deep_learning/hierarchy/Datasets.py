import torch
from torch.utils.data import Dataset, DataLoader

#################################################################################################################################
#################################################################################################################################

class NuToTrackDataset(Dataset):
    def __init__(self, input0, input1, y0, y1, y):
        # convert into PyTorch tensors and remember them
        self.input0 = torch.tensor(input0, dtype=torch.float32)
        self.input1 = torch.tensor(input1, dtype=torch.float32)
        self.y0 = torch.tensor(y0, dtype=torch.float32)
        self.y1 = torch.tensor(y1, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        # this should return the size of the dataset
        return len(self.y)
    
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features0 = self.input0[idx]
        features1 = self.input1[idx]
        target0 = self.y0[idx]
        target1 = self.y1[idx]
        target = self.y[idx]
        return features0, features1, target0, target1, target
    
#################################################################################################################################
#################################################################################################################################

class TrackToShowerDataset(Dataset):
    def __init__(self, input0, input1, y0, y1, y, trueGen):
        # convert into PyTorch tensors and remember them
        self.input0 = torch.tensor(input0, dtype=torch.float32)
        self.input1 = torch.tensor(input1, dtype=torch.float32)
        self.y0 = torch.tensor(y0, dtype=torch.float32)
        self.y1 = torch.tensor(y1, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.trueGen = torch.tensor(trueGen, dtype=torch.int)
        
    def __len__(self):
        # this should return the size of the dataset
        return len(self.y)
    
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features0 = self.input0[idx]
        features1 = self.input1[idx]
        target0 = self.y0[idx]
        target1 = self.y1[idx]
        target = self.y[idx]
        trueGen = self.trueGen[idx]
        return features0, features1, target0, target1, target, trueGen
    
#################################################################################################################################
#################################################################################################################################

class TrackToTrackDataset(Dataset):
    def __init__(self, input0, input1, input2, input3, y0, y1, y2, y3, y, trueGen):
        # convert into PyTorch tensors and remember them
        self.input0 = torch.tensor(input0, dtype=torch.float32)
        self.input1 = torch.tensor(input1, dtype=torch.float32)
        self.input2 = torch.tensor(input2, dtype=torch.float32)
        self.input3 = torch.tensor(input3, dtype=torch.float32)        
        self.y0 = torch.tensor(y0, dtype=torch.float32)
        self.y1 = torch.tensor(y1, dtype=torch.float32)
        self.y2 = torch.tensor(y2, dtype=torch.float32)
        self.y3 = torch.tensor(y3, dtype=torch.float32)        
        self.y = torch.tensor(y, dtype=torch.float32)
        self.trueGen = torch.tensor(trueGen, dtype=torch.int)
        
    def __len__(self):
        # this should return the size of the dataset
        return len(self.y)
    
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features0 = self.input0[idx]
        features1 = self.input1[idx]
        features2 = self.input2[idx]
        features3 = self.input3[idx]        
        target0 = self.y0[idx]
        target1 = self.y1[idx]
        target2 = self.y2[idx]
        target3 = self.y3[idx]        
        target = self.y[idx]
        trueGen = self.trueGen[idx]
        return features0, features1, features2, features3, target0, target1, target2, target3, target, trueGen
    
#################################################################################################################################
#################################################################################################################################