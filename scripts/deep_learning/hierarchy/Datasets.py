import torch
from torch.utils.data import Dataset, DataLoader

#################################################################################################################################
#################################################################################################################################

class TwoEdgeDataset(Dataset):
    def __init__(self, vars_edge0, vars_edge1, truth_edge0, truth_edge1, truth_link, truth_gen):
        self.vars_edge0 = torch.tensor(vars_edge0, dtype=torch.float32)
        self.vars_edge1 = torch.tensor(vars_edge1, dtype=torch.float32)
        self.truth_edge0 = torch.tensor(truth_edge0, dtype=torch.long)
        self.truth_edge1 = torch.tensor(truth_edge1, dtype=torch.long)
        self.truth_link = torch.tensor(truth_link, dtype=torch.float32)
        self.truth_gen = torch.tensor(truth_gen, dtype=torch.int)
        
    def __len__(self):
        return len(self.truth_link)
    
    def __getitem__(self, idx):
        vars_edge0 = self.vars_edge0[idx]
        vars_edge1 = self.vars_edge1[idx]
        truth_edge0 = self.truth_edge0[idx]
        truth_edge1 = self.truth_edge1[idx]
        truth_link = self.truth_link[idx]
        truth_gen = self.truth_gen[idx]
        return {"edge0":(vars_edge0, truth_edge0), "edge1":(vars_edge1, truth_edge1), "truth_link":truth_link, "truth_gen":truth_gen}
    
#################################################################################################################################
#################################################################################################################################

class FourEdgeDataset(Dataset):
    def __init__(self, vars_edge0, vars_edge1, vars_edge2, vars_edge3, truth_edge0, truth_edge1, truth_edge2, truth_edge3, truth_link, truth_gen):
        self.vars_edge0 = torch.tensor(vars_edge0, dtype=torch.float32)
        self.vars_edge1 = torch.tensor(vars_edge1, dtype=torch.float32)
        self.vars_edge2 = torch.tensor(vars_edge2, dtype=torch.float32)
        self.vars_edge3 = torch.tensor(vars_edge3, dtype=torch.float32)        
        self.truth_edge0 = torch.tensor(truth_edge0, dtype=torch.long)
        self.truth_edge1 = torch.tensor(truth_edge1, dtype=torch.long)
        self.truth_edge2 = torch.tensor(truth_edge2, dtype=torch.long)
        self.truth_edge3 = torch.tensor(truth_edge3, dtype=torch.long)        
        self.truth_link = torch.tensor(truth_link, dtype=torch.float32)
        self.truth_gen = torch.tensor(truth_gen, dtype=torch.int)
        
    def __len__(self):
        return len(self.truth_link)
    
    def __getitem__(self, idx):
        vars_edge0 = self.vars_edge0[idx]
        vars_edge1 = self.vars_edge1[idx]
        vars_edge2 = self.vars_edge2[idx]
        vars_edge3 = self.vars_edge3[idx]        
        truth_edge0 = self.truth_edge0[idx]
        truth_edge1 = self.truth_edge1[idx]
        truth_edge2 = self.truth_edge2[idx]
        truth_edge3 = self.truth_edge3[idx]        
        truth_link = self.truth_link[idx]
        truth_gen = self.truth_gen[idx]
        return {"edge0":(vars_edge0, truth_edge0), "edge1":(vars_edge1, truth_edge1), "edge2":(vars_edge2, truth_edge2), \
                "edge3":(vars_edge3, truth_edge3), "truth_link":truth_link, "truth_gen":truth_gen}

    
#################################################################################################################################
#################################################################################################################################