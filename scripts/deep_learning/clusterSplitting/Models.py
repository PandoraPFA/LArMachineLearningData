import torch
import torch.nn as nn
import math

################################################################################################################################################
################################################################################################################################################
 # Classification Model   
################################################################################################################################################
################################################################################################################################################          
    
class ConvModel(nn.Module):

    def __init__(self, device, num_features):
        super(ConvModel, self).__init__()
        
        self.device = device  
        
        # --- Encoder ---
        self.features = nn.Sequential(
            self.conv_block(in_channels=(num_features+1), out_channels=16, device=device, pool=False),
            self.conv_block(in_channels=16, out_channels=32, device=device, pool=True),
            self.conv_block(in_channels=32, out_channels=64, device=device, pool=False),
            self.conv_block(in_channels=64, out_channels=128, device=device, pool=True),
            nn.AdaptiveAvgPool1d(1) 
        )  
        
        # --- Output ---
        # ATTN: No softmax!
        self.output = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3, device=device)
        )    
        
    # --- Class functions ---    
    def conv_block(self, in_channels, out_channels, device, n_convs=2, pool=True):
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
            #layers += [nn.MaxPool1d(kernel_size=2)]
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        b, l, f = x.shape
        
        # Create a column of shape (L,) with normalized positions
        pos = torch.linspace(0, 1, steps=l, device=x.device)  # (L,)
        pos = pos.unsqueeze(0).repeat(b, 1)                   # (B, L)
        pos = pos.unsqueeze(2)                                # (B, L, 1)    
        x = torch.cat([x, pos], dim=2)  # (B, L, F+1)

        # Change (B, L, F) -> (B, F, L)
        x = x.permute(0, 2, 1)  
        
        # Create a sequence of angle and length and combine
        # angle_std = angle_std.unsqueeze(1).unsqueeze(1)
        # angle_std = torch.repeat_interleave(angle_std, x.shape[2], dim=2)
        # length = length.unsqueeze(1).unsqueeze(1)
        # length = torch.repeat_interleave(length, x.shape[2], dim=2)         
        # x = torch.cat([x, angle_std, length], dim=1)        
        
        # Run through model
        x = self.features(x)
        
        x = x.squeeze(2)
        # angle_std = angle_std.unsqueeze(1)
        # length = length.unsqueeze(1)
        # combined = torch.cat([x, angle_std, length], dim=1)
        
        output = self.output(x)

        return output     
    
################################################################################################################################################
################################################################################################################################################
 # Classification Model   
################################################################################################################################################
################################################################################################################################################

class ConvEncoderDecoder(nn.Module):
    def __init__(self, device, num_features):
        super().__init__()
        self.device = device  

        # --- Encoder ---
        self.enc1 = self.conv_block(num_features + 1, 16, device)
        self.enc2 = self.conv_block(16, 32, device)
        self.enc3 = self.conv_block(32, 64, device)
        self.enc4 = self.conv_block(64, 128, device)

        self.pool = nn.MaxPool1d(2)

        # --- Decoder ---
        self.up1 = self.up_block(128, 64, device)
        self.dec1 = self.conv_block(128, 64, device, n_convs=1)
        self.up2 = self.up_block(64, 32, device)
        self.dec2 = self.conv_block(64, 32, device, n_convs=1)
        self.up3 = self.up_block(32, 16, device)
        self.dec3 = self.conv_block(32, 16, device, n_convs=1)

        # Final output
        self.final = nn.Conv1d(16, 1, kernel_size=1, device=device)
        
        # self.final = nn.Sequential(
        #     nn.Conv1d(16+2, 8, kernel_size=1, device=device),
        #     nn.ReLU(),
        #     nn.Conv1d(8, 1, kernel_size=1, device=device))

    # --- Class functions ---
    def conv_block(self, in_channels, out_channels, device, n_convs=2):
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv1d(in_channels if i == 0 else out_channels,
                          out_channels, kernel_size=3, padding=1, device=device),
                nn.BatchNorm1d(out_channels, device=device),
                nn.ReLU()
            ]
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels, device):
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, device=device)

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
        #e4 = self.enc4(self.pool(e3))

        # Decoder
        #d1 = self.dec1(torch.cat([self.up1(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))
        
        # # Final output
        # angle_std = angle_std.unsqueeze(1).unsqueeze(1)
        # angle_std = torch.repeat_interleave(angle_std, d3.shape[2], dim=2)        
        # length = length.unsqueeze(1).unsqueeze(1)
        # length = torch.repeat_interleave(length, d3.shape[2], dim=2)         
        # combine = torch.cat([d3, angle_std, length], dim=1)
        
        # Output
        out = self.final(d3)                 # (B, 1, L)
        out = out.permute(0, 2, 1)           # (B, L, 1)        
        
        return out