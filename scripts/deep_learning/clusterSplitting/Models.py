import torch
import torch.nn as nn
import math

################################################################################################################################################
################################################################################################################################################
 # Helpers   
################################################################################################################################################
################################################################################################################################################  

def conv_block(in_channels, out_channels, device, n_convs=2, pool=False):
    layers = []
    for i in range(n_convs):
        layers += [
            nn.Conv1d(in_channels if i == 0 else out_channels,
                      out_channels, kernel_size=3, padding=1, device=device),
            nn.BatchNorm1d(out_channels, device=device),
            nn.ReLU()
        ]
    if pool:
        layers += [nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, device=device)] # Learn downsample

    return nn.Sequential(*layers)

################################################################################################################################################
################################################################################################################################################

def up_block(in_channels, out_channels, device):
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, device=device)

################################################################################################################################################
################################################################################################################################################
 # Classification Model   
################################################################################################################################################
################################################################################################################################################          
    
class ConvModel(nn.Module):
    """
    Model used to predict the nature (track not contaminated/track contaminated/shower) of a window within a cluster

    num_features: the number of time-series sequences
    """
    def __init__(self, device, num_features):
        super(ConvModel, self).__init__()
        
        self.device = device  
        
        # --- Encoder ---
        self.features = nn.Sequential(
            # position sequence adds an extra feature 
            conv_block(in_channels=(num_features+1), out_channels=16, device=device, pool=False),
            conv_block(in_channels=16, out_channels=32, device=device, pool=True),
            conv_block(in_channels=32, out_channels=64, device=device, pool=False),
            conv_block(in_channels=64, out_channels=128, device=device, pool=True),
            nn.AdaptiveAvgPool1d(1) 
        )  
        
        # --- Output ---
        # ATTN: No softmax!
        self.output = nn.Sequential(
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3, device=device)
        )    
        
    def forward(self, x):
        b, l, f = x.shape
        
        # Create a column of shape (L,) with normalized positions
        pos = torch.linspace(0, 1, steps=l, device=x.device)  # (L,)
        pos = pos.unsqueeze(0).repeat(b, 1)                   # (B, L)
        pos = pos.unsqueeze(2)                                # (B, L, 1)    
        x = torch.cat([x, pos], dim=2)  # (B, L, F+1)

        # Change (B, L, F) -> (B, F, L)
        x = x.permute(0, 2, 1)  
        
        # Run through model
        x = self.features(x)
        x = x.squeeze(2)
        
        output = self.output(x)

        return output     
    
################################################################################################################################################
################################################################################################################################################
 # Classification Model   
################################################################################################################################################
################################################################################################################################################

class ConvEncoderDecoder(nn.Module):
    """
    Model used to predict the nature (true/false split position) of each sequence element

    num_features: the number of time-series sequences
    """
    def __init__(self, device, num_features):
        super().__init__()
        self.device = device  

        # --- Encoder ---
        # position sequence adds an extra feature 
        self.enc1 = conv_block(num_features + 1, 16, device)
        self.enc2 = conv_block(16, 32, device)
        self.enc3 = conv_block(32, 64, device)

        self.pool = nn.MaxPool1d(2)

        # --- Decoder ---
        self.up1 = up_block(64, 32, device)
        self.dec1 = conv_block(64, 32, device, n_convs=1)
        self.up2 = up_block(32, 16, device)
        self.dec2 = conv_block(32, 16, device, n_convs=1)

        # Final output
        self.final = nn.Conv1d(16, 1, kernel_size=1, device=device)

    # --- Forward pass ---
    def forward(self, x):
        b, l, f = x.shape

        # Add in position info
        pos = torch.linspace(0, 1, steps=l, device=x.device)  # (L,)
        pos = pos.unsqueeze(0).repeat(b, 1)                   # (B, L)
        pos = pos.unsqueeze(2)                                # (B, L, 1)    
        x = torch.cat([x, pos], dim=2)                        # (B, L, F+1)   
        
        # Change (B, L, F) -> (B, F, L)
        x = x.permute(0, 2, 1)          

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
                
        # Output
        out = self.final(d2)                 # (B, 1, L)
        out = out.permute(0, 2, 1)           # (B, L, 1)        
        
        return out