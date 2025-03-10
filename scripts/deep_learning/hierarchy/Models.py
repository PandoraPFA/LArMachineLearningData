from torch import nn
import torch
import numpy as np

##########################################################################################################################
##########################################################################################################################

primary_track_n_links = 2
primary_shower_n_links = 1
primary_tier_n_orientation_indep_vars = 1
primary_tier_n_orientation_dep_vars = 8

track_track_n_links = 4
track_shower_n_links = 2
later_tier_n_orientation_indep_vars = 5
later_tier_n_orientation_dep_vars = 21

##########################################################################################################################
##########################################################################################################################

class MLPModel(nn.Module):

    def __init__(self, n_variables, dropoutRate=0.5):
        super(MLPModel, self).__init__()
        
        # first hidden layer
        self.linear1 = nn.Linear(n_variables, 128)
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

    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.dropout1(x)
        x = self.act2(self.linear2(x))
        x = self.dropout2(x)
        x = self.act3(self.linear3(x))
        x = self.dropout3(x)        
        x = self.act4(self.linear4(x))        
        x = self.dropout4(x)        
        return x

##########################################################################################################################
##########################################################################################################################

class PrimaryTrackShowerModel(nn.Module):

    def __init__(self, n_variables, dropoutRate=0.5):
        super(PrimaryTrackShowerModel, self).__init__()
        
        # MLPModel
        self.MLPModel = MLPModel(n_variables, dropoutRate)        
        # Output layer
        self.output = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.output.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.MLPModel.forward(x)      
        x = self.sigmoid(self.output(x))
        return x

##########################################################################################################################
##########################################################################################################################

class OrientationModel(nn.Module):
    
    def __init__(self, n_variables, dropoutRate=0.5):
        super(OrientationModel, self).__init__()
        # MLPModel
        self.MLPModel = MLPModel(n_variables, dropoutRate)        
        # Output layer
        self.output = nn.Linear(32, 3)
        nn.init.xavier_uniform_(self.output.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.MLPModel.forward(x)      
        x = self.softmax(self.output(x))
        return x
    
##########################################################################################################################
##########################################################################################################################

class ClassifierModel(nn.Module):

    def __init__(self, n_links):
        super(ClassifierModel, self).__init__()
        # Input layer and output
        self.output = nn.Linear((n_links * 3), 1)
        nn.init.xavier_uniform_(self.output.weight)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): 
        x = self.sigmoid(self.output(x))
        return x    
    
##########################################################################################################################
##########################################################################################################################

def PrepareBranchModelInput(n_links, n_global_vars, n_link_vars, variables) :

    i_end_of_global_vars = n_global_vars
    i_target_start = variables.shape[1] - n_link_vars
    i_rest_start = n_global_vars
    i_rest_end = n_global_vars + (n_link_vars * (n_links - 1))
    
    branch_model_input = [variables]
    
    for i in range(1, n_links) :
        this_input = np.concatenate((branch_model_input[-1][:,0:i_end_of_global_vars], branch_model_input[-1][:,i_target_start:], \
                                     branch_model_input[-1][:,i_rest_start:i_rest_end]), axis=1)
        
        branch_model_input.append(this_input)

    return branch_model_input

##########################################################################################################################
##########################################################################################################################

def GetClassificationScore(branch_model, classifier_model, n_links, n_orientation_indep_vars, n_orientation_dep_vars, variables):
    
    branch_model_input = PrepareBranchModelInput(n_links, n_orientation_indep_vars, n_orientation_dep_vars, variables)

    with torch.no_grad():
        branch_model.eval()
        classifier_model.eval()

        classifier_input = torch.empty(0,)

        for i in range(n_links) :
            pred = branch_model(torch.tensor(branch_model_input[i], dtype=torch.float))
            classifier_input = torch.concatenate((classifier_input, pred), axis=1)

        scores = classifier_model(classifier_input).reshape(-1).tolist()
    return scores



