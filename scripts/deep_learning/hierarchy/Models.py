from torch import nn
import torch

###################################################################################################################
# PrimaryTrackShowerModel
#################################################################################################################################

class PrimaryTrackShowerModel(nn.Module):

    def __init__(self, nVariables, dropoutRate=0.5):
        super(PrimaryTrackShowerModel, self).__init__()
        
        # first hidden layer
        self.linear1 = nn.Linear(nVariables, 128)
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropoutRate)
        # second hidden layer
        self.linear2 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()      
        self.dropout2 = nn.Dropout(dropoutRate)
        # third hidden layer
        self.linear3 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropoutRate)
        # forth hidden layer
        self.linear4 = nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.linear4.weight, nonlinearity='relu')
        self.act4 = nn.ReLU()             
        self.dropout4 = nn.Dropout(dropoutRate)
        # fifth hidden layer and output
        self.output = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.output.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.act1(self.linear1(x))
        x = self.dropout1(x)
        x = self.act2(self.linear2(x))
        x = self.dropout2(x)
        x = self.act3(self.linear3(x))
        x = self.dropout3(x)        
        x = self.act4(self.linear4(x))        
        x = self.dropout4(x)        
        x = self.sigmoid(self.output(x))
        return x

##########################################################################################################################
# OrientationModel
##########################################################################################################################

class OrientationModel(nn.Module):

    def __init__(self, nVariables, dropoutRate=0.5):
        super(OrientationModel, self).__init__()
        
        # first hidden layer
        self.linear1 = nn.Linear(nVariables, 128)
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropoutRate)
        # second hidden layer
        self.linear2 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()      
        self.dropout2 = nn.Dropout(dropoutRate)
        # third hidden layer
        self.linear3 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropoutRate)
        # forth hidden layer
        self.linear4 = nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.linear4.weight, nonlinearity='relu')
        self.act4 = nn.ReLU()             
        self.dropout4 = nn.Dropout(dropoutRate)
        # fifth hidden layer and output
        self.output = nn.Linear(32, 3)
        nn.init.xavier_uniform_(self.output.weight)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.dropout1(x)
        x = self.act2(self.linear2(x))
        x = self.dropout2(x)
        x = self.act3(self.linear3(x))
        x = self.dropout3(x)        
        x = self.act4(self.linear4(x))        
        x = self.dropout4(x)        
        x = self.softmax(self.output(x))
        return x
    
##########################################################################################################################
# Classifier Model
##########################################################################################################################

class ClassifierModel(nn.Module):

    def __init__(self, nInputs):
        super(ClassifierModel, self).__init__()
        
        # Input layer and output
        self.output = nn.Linear(nInputs, 1)
        nn.init.xavier_uniform_(self.output.weight)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): 
        x = self.sigmoid(self.output(x))
       # x = self.output(x)
        return x    